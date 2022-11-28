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
                    weight_decay=0, quad_model_adaptation_thresh=0, differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                    weight_decay=0, quad_model_adaptation_thresh=quad_model_adaptation_thresh, differentiable=differentiable)
        self.step_counter = 0
        self.old_step = None 

        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, 
            curvature: FisherInfo,
            curvature_kwargs: Dict,
            use_adaptive_learning_rate: bool = False,
            use_adaptive_momentum: bool = False,
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

        step, loss, output = ngd(
            params_with_grad=params_with_grad,
            curvature=curvature,
            curvature_kwargs=curvature_kwargs,
            lr=group['lr'],
            momentum=group['momentum'],
            weight_decay=group['weight_decay'],
            quad_model_adaptation_thresh=group['quad_model_adaptation_thresh'],
            use_adaptive_learning_rate=use_adaptive_learning_rate,
            use_adaptive_momentum=use_adaptive_momentum,
            use_adaptive_damping=use_adaptive_damping,
            use_approximate_quad_model=use_approximate_quad_model, 
            old_step=self.old_step,
            step_counter=self.step_counter,
            closure=closure
            )

        self.step_counter += 1
        self.old_step = step

        return loss, output

def ngd(params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        old_step: Tensor,
        lr: float = 0.1,
        momentum: float = 0.,
        weight_decay: float = 0.,
        quad_model_adaptation_thresh: float = 1e-6,
        use_adaptive_learning_rate: bool = False,
        use_adaptive_momentum: bool = False,
        use_adaptive_damping: bool = False,
        use_approximate_quad_model: bool = False, 
        min_damping: float = 1e-8,
        max_damping: float = 100.,
        damping_adaptation_interval: int = 5, 
        damping_adaptation_decay: float = 0.9,
        damping_lower_threshold: float = 0.25,
        damping_upper_threshold: float = 0.75,
        step_counter: int = 0, 
        closure=None
        ):

    func = _single_tensor_ngd

    step, loss, output = func(
        params_with_grad=params_with_grad,
        curvature=curvature,
        curvature_kwargs=curvature_kwargs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        quad_model_adaptation_thresh=quad_model_adaptation_thresh,
        use_adaptive_learning_rate=use_adaptive_learning_rate,
        use_adaptive_momentum=use_adaptive_momentum,
        use_adaptive_damping=use_adaptive_damping,
        use_approximate_quad_model=use_approximate_quad_model,
        min_damping=min_damping, 
        max_damping=max_damping,
        damping_adaptation_interval=damping_adaptation_interval, 
        damping_adaptation_decay=damping_adaptation_decay,
        damping_lower_threshold=damping_lower_threshold,
        damping_upper_threshold=damping_upper_threshold,
        old_step=old_step,
        step_counter=step_counter,
        closure=closure
    )

    return step, loss, output

def _single_tensor_ngd(
        params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        old_step: Tensor,
        momentum: float = 0.,
        lr: float = 0.1,
        weight_decay: float = 0.,#
        quad_model_adaptation_thresh: float = 1e-6,
        use_adaptive_learning_rate: bool = False,
        use_adaptive_momentum: bool = False,
        use_adaptive_damping: bool = False,
        use_approximate_quad_model: bool = False, 
        min_damping: float = 0,
        max_damping: float = 100.,
        damping_adaptation_interval: int = 5,
        damping_adaptation_decay: float = 0.9,
        damping_lower_threshold: float = 0.25,
        damping_upper_threshold: float = 0.75,
        closure = None,
        step_counter: int = 0,
        ):

    # update curvature estimate
    curvature.update(**curvature_kwargs)

    # compute loss and proposed directions (i.e. gradients: ∇h(θ = γ(c))) 
    with torch.enable_grad():
        loss, output = closure(parameters_vec=params_with_grad)
        loss.backward()
    
    descent_directions = params_with_grad.grad.detach() # ∇h
    # compute proposed directions (i.e. natrual gradients) -∆ = \tilde{F}^-1∇h
    # a.k.a. preconditioned gradients 
    natural_descent_directions = curvature.ema_cvp(
        descent_directions,
        include_damping=True, 
        include_Tikhonov_regularization=True,
        weight_decay=weight_decay,
        use_inverse=True # \tilde{F}^-1
    )

    # vectors = (natural_descent_directions, step) -> preconditioned_gradients, velocities
    # compute the optimal coefficients (αt learning rate and μt momentum coefficients)

    if old_step is None: 
        old_step = torch.zeros_like(natural_descent_directions) + 1e-6
    
    use_quad_model_adaptation = False
    if torch.abs(descent_directions.T @ old_step) < quad_model_adaptation_thresh:
        use_quad_model_adaptation = True

    if use_adaptive_learning_rate and use_quad_model_adaptation:

        lr, momentum = _compute_the_optimal_coefficients_via_quad_model(
            curvature=curvature,
            descent_directions=descent_directions,
            natural_descent_directions=natural_descent_directions,
            use_adaptive_momentum=use_adaptive_momentum,
            old_step=old_step, 
            weight_decay=weight_decay, 
        )
    
    # compute delta and return velocities, old_step, a.k.a 𝛿 = -lr*F^-1 ∇h + μ*v
    step = -lr*natural_descent_directions + momentum * old_step
    params_with_grad.add_(step, alpha=1) # update parameters c + 𝛿

    # Optionally compute the reduction ratio and update the damping
    if (use_adaptive_damping and 
            ((step_counter + 1) % damping_adaptation_interval == 0)):

        adaptive_damping_kwargs = {
            'weight_decay': weight_decay,
            'damping_adaptation_interval': damping_adaptation_interval,
            'damping_adaptation_decay': damping_adaptation_decay, 
            'damping_lower_threshold': damping_lower_threshold,
            'damping_upper_threshold': damping_upper_threshold,
            'min_damping': min_damping, 
            'max_damping': max_damping,
            'use_quad_model_adaptation': use_quad_model_adaptation,
            'use_approximate_quad_model': False
            }
        
        damping = _compute_new_damping_and_rho(
            curvature=curvature,
            closure=closure,
            old_loss=loss,
            params_with_grad=params_with_grad,
            step=step,
            **adaptive_damping_kwargs
        )
        curvature.curvature_damping.damping = damping

        print({f'lr: {lr}'})
        print({f'momentum: {momentum}'})
    
    return step, loss.detach(), output.detach()

def _compute_the_optimal_coefficients_via_quad_model(
        curvature: FisherInfo, 
        natural_descent_directions: Tensor, 
        descent_directions: Tensor,
        old_step: Tensor,
        use_adaptive_momentum: bool = True,
        weight_decay: float = 0., 
        use_approximate_quad_model: bool = False
    ):

    regulariser = curvature.curvature_damping.damping + weight_decay
    Δ = - natural_descent_directions
    JcΔ = curvature.exact_cvp(
        Δ, use_square_root=True
        ).flatten() if not use_approximate_quad_model else curvature.ema_cvp(
            Δ, use_square_root=True
            )
    ΔTΔ = Δ.T @ Δ
    ΔTFΔ = JcΔ.T @ JcΔ + (ΔTΔ * regulariser)
    if use_adaptive_momentum:
        Jcold_step = curvature.exact_cvp(
            old_step, use_square_root=True
            ).flatten() if not use_approximate_quad_model else curvature.ema_cvp(
                Δ, use_square_root=True
                )
        old_stepTold_step = old_step.T @ old_step
        old_stepTFold_step = Jcold_step.T @ Jcold_step + old_stepTold_step*regulariser
        ΔTFold_step = JcΔ.T @ Jcold_step + regulariser*Δ.T@old_step
        matrix = torch.Tensor([[ΔTFΔ, ΔTFold_step],[ΔTFold_step, old_stepTFold_step]])
        b = torch.Tensor([descent_directions.T@Δ, descent_directions.T@old_step])
        optimal_coeffs = torch.linalg.solve(-matrix, b)
    else:
        optimal_coeffs = torch.Tensor([- Δ.T @ descent_directions / (ΔTFΔ), 0.])
    assert optimal_coeffs.shape == (2,)

    return optimal_coeffs[0],  optimal_coeffs[1]

def _solve_quad_model(
        curvature: FisherInfo,
        descent_directions: Tensor, # F^-1 ∇h 
        step: Tensor,
        weight_decay: float,
        use_approximate_quad_model: bool = False
    ) -> Tuple[Tensor, Tensor]:
    
    regulariser = curvature.curvature_damping.damping + weight_decay
    Jc𝛿 = curvature.exact_cvp(step,
            use_square_root=True
                ).flatten() if not use_approximate_quad_model else curvature.ema_cvp(step,
                    use_square_root=True
                    )

    𝛿TF𝛿 = Jc𝛿.T @ Jc𝛿 + step.T@step * regulariser
    g = descent_directions.T @ step

    return 𝛿TF𝛿, g

def _compute_quadratic_model_value(
        curvature: FisherInfo,
        descent_directions: Tensor,
        step: Tensor,
        weight_decay: float,
        use_approximate_quad_model: bool = False
    ) -> Tensor:

    𝛿TF𝛿, g = _solve_quad_model(
        curvature=curvature, 
        descent_directions=descent_directions, 
        step=step, 
        weight_decay=weight_decay,
        use_approximate_quad_model=use_approximate_quad_model
        )

    return 𝛿TF𝛿 / 2 + g

def _compute_new_damping_and_rho(
        curvature: FisherInfo, 
        closure: callable, 
        old_loss: Tensor, 
        params_with_grad: Tensor,
        step: Tensor,
        weight_decay: float,
        min_damping: float = 1e-8,
        max_damping: float = 100.,
        damping_adaptation_interval: int = 5, 
        damping_adaptation_decay: float = 0.95,
        damping_lower_threshold: float = 0.25,
        damping_upper_threshold: float = 0.75,
        use_quad_model_adaptation: bool = False,
        use_approximate_quad_model: bool = False
    ) -> Tuple[float,float]:
    
    # reduction ratio
    # at this point params_with_grad have been updated but not the grads
    change = _compute_quadratic_model_value(
        curvature=curvature, 
        step=step, descent_directions=params_with_grad.grad, # ∇h
        weight_decay=weight_decay, use_approximate_quad_model=use_approximate_quad_model
    ) if use_quad_model_adaptation else params_with_grad.grad.T @ step # ∇h𝛿

    # at this point params_with_grad have been updated
    new_loss = closure(parameters_vec=params_with_grad)[0]
    change_in_objecvitve = new_loss - old_loss
    rho = (change_in_objecvitve / change).cpu().numpy()
    rho_not_nan = np.nan_to_num(rho, nan=-100.0)
    
    # update damping
    current_damping = curvature.curvature_damping.damping
    if rho_not_nan < damping_lower_threshold:
        damping = damping_adaptation_decay**(-damping_adaptation_interval)*current_damping
    elif rho_not_nan > damping_upper_threshold:
        damping = damping_adaptation_decay**(damping_adaptation_interval)*current_damping
    else:
        damping = current_damping
    damping = np.clip(damping, a_min=min_damping, a_max=max_damping)


    print(f'change: {change}')
    print(f'damping: {curvature.curvature_damping.damping}')

    return damping
