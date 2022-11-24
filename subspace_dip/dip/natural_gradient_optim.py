from typing import List, Tuple, Dict
import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from typing import List

from .fisher_info import FisherInfo

__all__ = ['NGD', 'ngd']

class _RequiredParameter(object):
    """ Singleton class representing a required parameter for an Optimizer.
        https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
    """
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

def _use_grad_for_differentiable(func):
    """ Singleton class representing a required parameter for an Optimizer.
        https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
    """
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad

class NGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False, differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                    weight_decay=0, nesterov=nesterov, differentiable=differentiable)
        self.step_counter = 0
        self.old_step = None 
        self.old_step_no_momentun = None

        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, 
            curvature: FisherInfo,
            curvature_kwargs: Dict, 
            use_adaptive_damping: bool = False,
            use_approximate_quad_model: bool = False,
            max_length_memory: int = 5,
            closure=None,
            ):
        
        """Performs a single optimization step.
        Args:
            curvature_matrix
        """

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params_with_grad = group['params'][0]

        old_step, old_step_no_momentun, loss, output = ngd(
            params_with_grad=params_with_grad,
            curvature=curvature,
            curvature_kwargs=curvature_kwargs,
            weight_decay=group['weight_decay'],
            lr=group['lr'],
            momentum=group['momentum'],
            use_adaptive_damping=use_adaptive_damping,
            use_approximate_quad_model=use_approximate_quad_model, 
            old_step=self.old_step,
            old_step_no_momentun=self.old_step_no_momentun,
            step_counter=self.step_counter,
            closure=closure
            )

        self.step_counter += 1
        self.old_step = old_step
        self.old_step_no_momentun = old_step_no_momentun

        return loss, output

def ngd(params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        weight_decay: float,
        lr: float,
        old_step: Tensor,
        old_step_no_momentun: Tensor, 
        momentum: float = 0.,
        use_adaptive_damping: bool = True,
        use_approximate_quad_model: bool = False, 
        min_damping: float = 1e-8,
        max_damping: float = 100.,
        damping_adaptation_interval: int = 5, 
        damping_adaptation_decay: float = 0.9,
        damping_lower_threshold: float = 0.25,
        damping_upper_threshold: float = 0.75,
        include_damping_in_quad_change: bool = False,
        step_counter: int = 0, 
        closure=None
        ):


    """Performs NGD algorithm computation.
        TODO:
        use_adaptive_learning_rate: Boolean. 
            Specifies whether the optimizer will use the quadratic model induced 
            by the true curvature matrix to automatically pick the learning rate 
            or it would be fixed.
            (Default: ``True``)
        use_adaptive_damping: Boolean. 
            Specifies whether the optimizer will use the Levenberg-Marquardt method 
            to try to adjust the damping automatically every ``damping_adaptation_interval`` 
            iterations. If this is set to ``False`` the damping is fixed to the 
            ``initial_damping`` set to initialise ``curvature``. The damping value 
            times the identity matrix is added to the curvature matrix (i.e. the Fisher) 
            every time the curvature (or its inverse) is multiplied with a vector 
            (``include_damping`` in the curvature method is defaulted to True). 
            (Default: ``True``)
        min_damping: Scalar. 
            Minimum value the damping parameter can take.
            (Default: ``1e-8``)
        max_damping: Scalar. Maximum value the damping parameter can take.
            (Default: ``Infinity``)
        include_damping_in_quad_change: Boolean. 
            Whether to include the contribution of the extra isotropic damping term 
            in the quadratic model value for the purposes computing the reduction ration
            (``rho``). This is only used when adapting the damping parameter. 
            Note that the extra damping from the ``l2_reg`` argument is always included.
            (Default: ``True``)
        """

    func = _single_tensor_ngd

    old_step, old_step_no_momentun, loss, output = func(
        params_with_grad=params_with_grad,
        curvature=curvature,
        curvature_kwargs=curvature_kwargs, 
        momentum=momentum,
        weight_decay=weight_decay,
        lr=lr,
        use_adaptive_damping=use_adaptive_damping,
        use_approximate_quad_model=use_approximate_quad_model,
        min_damping=min_damping, 
        max_damping=max_damping,
        damping_adaptation_interval=damping_adaptation_interval, 
        damping_adaptation_decay=damping_adaptation_decay,
        damping_lower_threshold=damping_lower_threshold,
        damping_upper_threshold=damping_upper_threshold,
        include_damping_in_quad_change=include_damping_in_quad_change,
        old_step=old_step,
        old_step_no_momentun=old_step_no_momentun,
        step_counter=step_counter,
        closure=closure
    )

    return old_step, old_step_no_momentun, loss, output

def _single_tensor_ngd(
        params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        old_step: Tensor,
        old_step_no_momentun: Tensor, 
        momentum: float = 0.,
        weight_decay: float = 0.,
        lr: float = 1e-3,
        use_adaptive_damping: bool = False,
        use_approximate_quad_model: bool = False, 
        min_damping: float = 1e-8,
        max_damping: float = 100.,
        damping_adaptation_interval: int = 5,
        damping_adaptation_decay: float = 0.9,
        damping_lower_threshold: float = 0.25,
        damping_upper_threshold: float = 0.75,
        damping_adaptation_start: int = 100, 
        include_damping_in_quad_change: bool = False,
        closure = None,
        step_counter: int = 0,
        ):

    # update curvature estimate
    curvature.update(
        **curvature_kwargs
    )

    # compute loss and proposed directions
    with torch.enable_grad():
        loss, output = closure(parameters_vec=params_with_grad)
        loss.backward()
    
    proposed_descent_directions = params_with_grad.grad.detach()
    natural_descent_directions = curvature.ema_cvp(
        proposed_descent_directions,
        weight_decay=weight_decay,
        use_inverse=True
    )
    
    # TODO:  compute the optimal coefficients

    # Update parameters
    if momentum !=0:
        if old_step is None:
            old_step = lr*torch.clone(natural_descent_directions).detach()
        else:
            old_step.mul_(momentum).add_(natural_descent_directions, alpha=lr)
        params_with_grad.add_(old_step, alpha=-1)
    else:
        old_step = lr*natural_descent_directions
        params_with_grad.add_(old_step, alpha=-1)

    # Optionally compute the reduction ratio and update the damping
    if ( use_adaptive_damping and 
            ((step_counter + 1) % damping_adaptation_interval == 0) 
                and step_counter > damping_adaptation_start):

        adaptive_damping_kwargs = {
            'weight_decay': weight_decay,
            'damping_adaptation_interval': damping_adaptation_interval,
            'damping_adaptation_decay': damping_adaptation_decay, 
            'damping_lower_threshold': damping_lower_threshold,
            'damping_upper_threshold': damping_upper_threshold,
            'min_damping': min_damping, 
            'max_damping': max_damping,
            'use_approximate_quad_model': use_approximate_quad_model,
            'include_damping_in_quad_change': include_damping_in_quad_change,
            }
        
        new_damping = _compute_new_damping_and_rho(
            curvature=curvature,
            closure=closure,
            old_loss=loss,
            params_with_grad=params_with_grad,
            old_step_no_momentun=lr*natural_descent_directions,
            **adaptive_damping_kwargs
        )
        curvature.damping = new_damping

    return old_step, lr*natural_descent_directions, loss.detach(), output.detach()

def _solve_quad_model(
        curvature: FisherInfo,
        proposed_descent_directions: Tensor,
        delta: Tensor,
        weight_decay: float, 
        include_damping_in_quad_change: bool = True,
        use_approximate_quad_model: bool = False
    ) -> Tuple[Tensor, Tensor]:

    if not use_approximate_quad_model:
        A = delta @ curvature.exact_cvp(
            delta,
            include_damping=include_damping_in_quad_change,
            weight_decay=weight_decay,
            use_inverse=False
        )
    else:
        A = delta @ curvature.ema_cvp(
            delta,
            include_damping=include_damping_in_quad_change,
            weight_decay=weight_decay,
            use_inverse=False
        )

    b = proposed_descent_directions @ delta
    return (A, b)
    
def _compute_quadratic_model_value(
        curvature: FisherInfo,
        proposed_descent_directions: Tensor,
        delta: Tensor,
        weight_decay: float, 
        include_damping_in_quad_change: bool = True,
        use_approximate_quad_model: bool = False
    ) -> Tensor:

    A, b = _solve_quad_model(
        curvature=curvature, 
        proposed_descent_directions=proposed_descent_directions, 
        delta=delta, 
        weight_decay=weight_decay, 
        include_damping_in_quad_change=include_damping_in_quad_change,
        use_approximate_quad_model=use_approximate_quad_model
        )

    return A / 2 + b

def _compute_new_damping_and_rho(
        curvature: FisherInfo, 
        closure: callable, 
        old_loss: Tensor, 
        params_with_grad: Tensor,
        old_step_no_momentun: Tensor,
        weight_decay: float,
        min_damping: float = 1e-8,
        max_damping: float = 100.,
        damping_adaptation_interval: int = 5, 
        damping_adaptation_decay: float = 0.95,
        damping_lower_threshold: float = 0.25,
        damping_upper_threshold: float = 0.75,
        include_damping_in_quad_change: bool = False,
        use_approximate_quad_model: bool = False
    ) -> Tuple[float,float]:
    
    # reduction ratio
    delta = -old_step_no_momentun
    quad_change = _compute_quadratic_model_value(
        curvature=curvature, delta=delta, proposed_descent_directions=params_with_grad.grad, 
        include_damping_in_quad_change=include_damping_in_quad_change, 
        weight_decay=weight_decay, use_approximate_quad_model=use_approximate_quad_model
    )

    next_params = params_with_grad + delta
    new_loss = closure(parameters_vec=next_params)[0]
    change_in_objecvitve = new_loss - old_loss
    rho = (change_in_objecvitve / quad_change).cpu().numpy()
    rho_not_nan = np.nan_to_num(rho, nan=-100.0)
    
    # update damping
    current_damping = curvature.damping
    if rho_not_nan < damping_lower_threshold:
        damping = damping_adaptation_decay**(-damping_adaptation_interval)*current_damping
    elif rho_not_nan > damping_upper_threshold:
        damping = damping_adaptation_decay**(damping_adaptation_interval)*current_damping
    else:
        damping = current_damping
    damping = np.clip(damping, a_min=min_damping, a_max=max_damping)

    print(rho_not_nan)
    print(change_in_objecvitve)
    print(quad_change)
    print(damping)

    return damping
