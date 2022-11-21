import torch
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
                    weight_decay=0,  differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay,
                        differentiable=differentiable)

        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, fisher_info: FisherInfo, closure=None, loss=None, use_adaptive_damping: bool = False):
        
        """Performs a single optimization step.
        Args:
            fisher_info_matrix
        """
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            ngd(params_with_grad,
                d_p_list,
                fisher_info,
                weight_decay=group['weight_decay'],
                lr=group['lr'],
                use_adaptive_damping=use_adaptive_damping,
                closure=closure, 
                loss=loss
                )

def ngd(params: List[Tensor],
        d_p_list: List[Tensor],
        fisher_info: FisherInfo,
        weight_decay: float,
        lr: float,
        use_adaptive_damping: bool = False,
        closure=None, 
        loss=None
        ):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    func = _single_tensor_ngd

    func(params,
        d_p_list,
        fisher_info=fisher_info,
        weight_decay=weight_decay,
        lr=lr,
        use_adaptive_damping=use_adaptive_damping, 
        closure=closure, 
        loss=loss
        )

def _compute_reduction_ratio(fisher_info, closure, loss, param, weight_decay, d_p, lr):

    delta = - lr * d_p
    den = .5 * delta @ fisher_info.fvp(delta, weight_decay=weight_decay) + param.grad @ delta
    up_params = param + delta
    next_loss = closure(parameters_vec=up_params)[0]
    num = next_loss - loss
    rho = num / den

    return rho.item()


def _single_tensor_ngd(params: List[Tensor],
        d_p_list: List[Tensor],
        fisher_info: FisherInfo,
        weight_decay: float,
        lr: float, 
        use_adaptive_damping: bool = False,
        closure=None, 
        loss=None
    ):

    for i, param in enumerate(params):
        d_p = d_p_list[i] 

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        d_p = fisher_info.fvp(
                d_p,
                weight_decay=weight_decay,
                use_inverse=True
            )
        
        if use_adaptive_damping: 
            rho = _compute_reduction_ratio(
                fisher_info=fisher_info,
                closure=closure, 
                loss=loss, 
                param=param,
                weight_decay=weight_decay,
                d_p=d_p, 
                lr=lr
            )

            if rho < 0.25:
                damping_fct = fisher_info.damping_fct
                fisher_info.damping_fct = (19/20)**(-1) * damping_fct
            elif rho > 0.75:
                damping_fct = fisher_info.damping_fct
                fisher_info.damping_fct = (19/20)**(1) * damping_fct

        d_p = fisher_info.fvp(
                d_p, 
                use_inverse=True
            )
        
        param.add_(d_p, alpha=-lr)
