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

    def __init__(self, params, lr=required, momentum=0, weight_decay=0,
            stats_interval=0, curvature_reduction_scale=0., differentiable=False):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=0, 
                    stats_interval=stats_interval, curvature_reduction_scale=curvature_reduction_scale, 
                    differentiable=differentiable
                )
        self.step_counter = 0
        self.old_step = None 
        self.FREEZE_DAMPING_UPDATE = False # switch activated when empirical Fisher low-rank regime is exited

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
            return_stats: bool = False
            ):
        
        """Performs a single optimization step.
        Args:
            curvature_matrix
        """

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params_with_grad = group['params'][0]

        step, loss, output, curvature_reduction_scale, FREEZE_DAMPING_UPDATE, stats = ngd(
            params_with_grad=params_with_grad,
            curvature=curvature,
            curvature_kwargs=curvature_kwargs,
            lr=group['lr'],
            momentum=group['momentum'],
            weight_decay=group['weight_decay'],
            stats_interval=group['stats_interval'],
            curvature_reduction_scale=group['curvature_reduction_scale'],
            use_adaptive_learning_rate=use_adaptive_learning_rate,
            use_adaptive_momentum=use_adaptive_momentum,
            use_adaptive_damping=use_adaptive_damping,
            use_approximate_quad_model=use_approximate_quad_model, 
            old_step=self.old_step,
            step_counter=self.step_counter,
            closure=closure,
            FREEZE_DAMPING_UPDATE=self.FREEZE_DAMPING_UPDATE,
            return_stats=return_stats
            )
        
        self.step_counter += 1
        self.old_step = step
        self.param_groups[0]['curvature_reduction_scale'] = curvature_reduction_scale
        self.FREEZE_DAMPING_UPDATE=FREEZE_DAMPING_UPDATE

        return loss, output, stats

def ngd(params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        old_step: Tensor,
        lr: float = 0.1,
        momentum: float = 0.,
        weight_decay: float = 0.,
        stats_interval: int = 20,
        curvature_reduction_scale: float = 1., 
        use_adaptive_learning_rate: bool = False,
        use_adaptive_momentum: bool = False,
        use_adaptive_damping: bool = False,
        use_approximate_quad_model: bool = False, 
        min_hyperparam: float = 1e-8,
        max_hyperparam: float = 100.,
        adaptation_interval: int = 5, 
        adaptation_decay: float = 0.75,
        lower_threshold: float = 0.25,
        upper_threshold: float = 0.75,
        closure=None,
        step_counter: int = 0, 
        FREEZE_DAMPING_UPDATE: bool = False,
        return_stats: bool = False
        ):

    func = _single_tensor_ngd

    outs = func(
        params_with_grad=params_with_grad,
        curvature=curvature,
        curvature_kwargs=curvature_kwargs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        stats_interval=stats_interval,
        curvature_reduction_scale=curvature_reduction_scale,
        use_adaptive_learning_rate=use_adaptive_learning_rate,
        use_adaptive_momentum=use_adaptive_momentum,
        use_adaptive_damping=use_adaptive_damping,
        use_approximate_quad_model=use_approximate_quad_model,
        min_hyperparam=min_hyperparam, 
        max_hyperparam=max_hyperparam,
        adaptation_interval=adaptation_interval, 
        adaptation_decay=adaptation_decay,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        old_step=old_step,
        closure=closure,
        step_counter=step_counter,
        FREEZE_DAMPING_UPDATE=FREEZE_DAMPING_UPDATE,
        return_stats=return_stats
    )

    return outs

def _single_tensor_ngd(
        params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        old_step: Tensor,
        momentum: float = 0.,
        lr: float = 0.1,
        weight_decay: float = 0.,
        stats_interval: int = 20,
        curvature_reduction_scale: float = 1., 
        use_adaptive_learning_rate: bool = False,
        use_adaptive_momentum: bool = False,
        use_adaptive_damping: bool = False,
        use_approximate_quad_model: bool = False, 
        min_hyperparam: float = 1e-8,
        max_hyperparam: float = 100.,
        max_lr: float = 5e2, 
        min_lr: float = - np.inf, 
        max_momentum: float = 1., 
        min_momentum: float = -np.inf, 
        adaptation_interval: int = 5,
        adaptation_decay: float = 0.75,
        lower_threshold: float = 0.25,
        upper_threshold: float = 0.75,
        closure = None,
        step_counter: int = 0,
        FREEZE_DAMPING_UPDATE: bool = False,
        return_stats: bool = False
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
    natural_descent_directions = curvature.approx_fisher_vp(
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
    
    if use_adaptive_learning_rate:

        lr, momentum = _compute_the_optimal_coefficients_via_quad_model(
            curvature=curvature,
            descent_directions=descent_directions,
            natural_descent_directions=natural_descent_directions,
            use_adaptive_momentum=use_adaptive_momentum,
            old_step=old_step, 
            weight_decay=weight_decay,
            curvature_reduction_scale=curvature_reduction_scale, 
            use_approximate_quad_model=use_approximate_quad_model
        )
        lr = np.clip(lr, a_min=min_lr, a_max=max_lr)
        momentum = np.clip(momentum, a_min=min_momentum, a_max=max_momentum)
    
    # compute delta and return velocities, old_step, a.k.a 𝛿 = -lr*F^-1 ∇h + μ*v
    step = -lr*natural_descent_directions + momentum * old_step
    params_with_grad.add_(step, alpha=1) # update parameters c + 𝛿

    # Optionally compute the reduction ratio and update the damping if not FREEZE_DAMPING_UPDATE
    if use_adaptive_damping and ((step_counter + 1) % adaptation_interval == 0 and not FREEZE_DAMPING_UPDATE):

        damping_adaptive_kwargs = {
            'weight_decay': weight_decay,
            'adaptation_interval': adaptation_interval,
            'adaptation_decay': adaptation_decay,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'min_hyperparam': min_hyperparam,
            'max_hyperparam': max_hyperparam,
            'use_approximate_quad_model': use_approximate_quad_model
            }
        current_damping = curvature.curvature_damping.damping
        updated_damping, rho, change = _update_hyperparam_based_on_reduction_ratio(
            curvature=curvature,
            closure=closure,
            old_loss=loss,
            params_with_grad=params_with_grad,
            step=step,
            current_hyperparam=current_damping, 
            curvature_reduction_scale=curvature_reduction_scale,
            **damping_adaptive_kwargs
        )
        curvature.curvature_damping.damping = updated_damping
        if updated_damping <= min_hyperparam:
            # empirical Fisher low-rank regime exited
            FREEZE_DAMPING_UPDATE = False #TODO: remove this flag is not needed 
    
    if (step_counter + 1) % adaptation_interval == 0 and use_adaptive_learning_rate:
        
        curvature_reduction_adaptive_kwargs = {
            'weight_decay': weight_decay,
            'adaptation_interval': adaptation_interval,
            'adaptation_decay': adaptation_decay,
            'lower_threshold': 0.75,
            'upper_threshold': 1.25,
            'use_reciprocal': False,
            'min_hyperparam': 1e-3,
            'max_hyperparam': 1.,
            'use_approximate_quad_model': use_approximate_quad_model
            }

        #TODO: optimise no need for double operation
        updated_curvature_reduction_scale, rho, change = _update_hyperparam_based_on_reduction_ratio(
            curvature=curvature,
            closure=closure,
            old_loss=loss,
            params_with_grad=params_with_grad,
            step=step,
            current_hyperparam=curvature_reduction_scale,
            curvature_reduction_scale=curvature_reduction_scale,
            **curvature_reduction_adaptive_kwargs
        )
        curvature_reduction_scale = updated_curvature_reduction_scale

    stats = None
    if return_stats and (step_counter + 1) % stats_interval == 0:
        stats = {
            'rho': rho if 'rho' in locals() else 0.,
            'model_change': change if 'change' in locals() else 0.,
            'curvature_damping': curvature.curvature_damping.damping,
            'lr': lr, 
            'momentum': momentum if hasattr(momentum, 'item') else momentum,
            'curvature_reduction_scale': curvature_reduction_scale, 
            'step': step.pow(2).sum().item(), 
            'descent_directions': descent_directions.pow(2).sum().item(),
            'natural_descent_directions_norm': natural_descent_directions.pow(2).sum().item()
        }

    outs = (step, loss.detach(), output.detach(), curvature_reduction_scale, FREEZE_DAMPING_UPDATE, None) if not return_stats \
                else (step, loss.detach(), output.detach(), curvature_reduction_scale, FREEZE_DAMPING_UPDATE, stats)
    
    return outs

def _compute_the_optimal_coefficients_via_quad_model(
        curvature: FisherInfo, 
        natural_descent_directions: Tensor, 
        descent_directions: Tensor,
        old_step: Tensor,
        use_adaptive_momentum: bool = True,
        weight_decay: float = 0., 
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 1., 
    ):

    assert old_step.ndim == 1
    assert old_step.ndim == natural_descent_directions.ndim
    assert old_step.ndim == descent_directions.ndim

    regulariser = curvature.curvature_damping.damping + weight_decay
    Δ = - natural_descent_directions
    if not use_approximate_quad_model: 
        JcΔ = curvature.exact_fisher_vp(Δ, use_square_root=True).flatten()  
    else: 
        JcΔ = curvature.approx_fisher_vp(Δ, use_square_root=True)

    ΔTΔ = torch.dot(Δ, Δ)
    ΔTFΔ = torch.dot(JcΔ, JcΔ) + (ΔTΔ * regulariser)
    if use_adaptive_momentum:
        if not use_approximate_quad_model: 
            Jcold_step = curvature.exact_fisher_vp(
                old_step, use_square_root=True
                ).flatten()  
        else: 
            Jcold_step = curvature.approx_fisher_vp(old_step, use_square_root=True)
        old_stepTold_step = torch.dot(old_step, old_step)
        old_stepTFold_step = torch.dot(Jcold_step, Jcold_step) + old_stepTold_step*regulariser
        ΔTFold_step = torch.dot(JcΔ, Jcold_step) + torch.dot(Δ, old_step) * regulariser
        sys_matrix = curvature_reduction_scale*torch.Tensor([[ΔTFΔ, ΔTFold_step],[ΔTFold_step, old_stepTFold_step]])
        b = torch.Tensor([torch.dot(descent_directions, Δ), torch.dot(descent_directions, old_step)])
        optimal_coeffs = torch.linalg.solve(-sys_matrix, b)
    else:
        optimal_coeffs = torch.Tensor([- torch.dot(descent_directions, Δ) / (curvature_reduction_scale*ΔTFΔ), 0.])
    assert optimal_coeffs.shape == (2,)

    return optimal_coeffs[0].item(),  optimal_coeffs[1].item()

def _get_quad_model(
        curvature: FisherInfo,
        descent_directions: Tensor, # F^-1 ∇h 
        step: Tensor,
        weight_decay: float,
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 1.
    ) -> Tuple[Tensor, Tensor]:
    
    assert step.ndim == 1
    assert descent_directions.ndim == step.ndim
    regulariser = curvature.curvature_damping.damping + weight_decay
    Jc𝛿 = curvature.exact_fisher_vp(step,
            use_square_root=True
                ).flatten() if not use_approximate_quad_model else curvature.approx_fisher_vp(step,
                    use_square_root=True
                    )
    assert Jc𝛿.ndim == 1
    𝛿TF𝛿 = torch.dot(Jc𝛿, Jc𝛿) + torch.dot(step, step) * regulariser
    tangent_plane = torch.dot(descent_directions, step)

    return curvature_reduction_scale*𝛿TF𝛿, tangent_plane

def _compute_quadratic_model_value(
        curvature: FisherInfo,
        descent_directions: Tensor,
        step: Tensor,
        weight_decay: float,
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 1.
    ) -> Tensor:

    scaled_𝛿TF𝛿, tangent_plane = _get_quad_model(
        curvature=curvature, 
        descent_directions=descent_directions, 
        step=step, 
        weight_decay=weight_decay,
        use_approximate_quad_model=use_approximate_quad_model, 
        curvature_reduction_scale=curvature_reduction_scale
        )
    quad_model_change = scaled_𝛿TF𝛿/2 + tangent_plane
    return quad_model_change.item(), scaled_𝛿TF𝛿, tangent_plane

def _update_hyperparam_based_on_reduction_ratio(
        curvature: FisherInfo,
        closure: callable,
        old_loss: Tensor,
        params_with_grad: Tensor,
        step: Tensor,
        current_hyperparam: float, 
        weight_decay: float,
        min_hyperparam: float = 1e-0,
        max_hyperparam: float = 100.,
        adaptation_interval: int = 5, 
        adaptation_decay: float = 0.75,
        lower_threshold: float = 0.25,
        upper_threshold: float = 0.75,
        ratio_lower_threshold: float = 0.9, 
        ratio_upper_threshold: float = 1.01, 
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 0.001, 
        use_reciprocal: bool = False
    ) -> Tuple[float,float]:
    
    reciprocal = - 1 if use_reciprocal else 1

    # reduction ratio
    # at this point params_with_grad have been updated but not the grads
    change, scaled_𝛿TF𝛿, tangent_plane = _compute_quadratic_model_value(
        curvature=curvature,
        step=step, descent_directions=params_with_grad.grad, # ∇h
        weight_decay=weight_decay, use_approximate_quad_model=use_approximate_quad_model,
        curvature_reduction_scale=curvature_reduction_scale)

    # at this point params_with_grad have been updated
    new_loss = closure(parameters_vec=params_with_grad)[0]
    change_in_objecvitve = new_loss - old_loss
    rho = (change_in_objecvitve / change).cpu().numpy()
    rho_not_nan = np.nan_to_num(rho, nan=-100.0)

    if rho_not_nan < lower_threshold:
        updated_hyperparam = adaptation_decay**(-adaptation_interval*reciprocal)*current_hyperparam
    elif rho_not_nan > upper_threshold:
        updated_hyperparam = adaptation_decay**(adaptation_interval*reciprocal)*current_hyperparam
    else:
        updated_hyperparam = current_hyperparam
    updated_hyperparam = np.clip(updated_hyperparam, a_min=min_hyperparam, a_max=max_hyperparam)

    return updated_hyperparam, rho_not_nan, change
