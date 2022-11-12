import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List

from .fisher_info import FisherInfoMat

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
    def step(self, fisher_info_matrix: FisherInfoMat, closure=None):
        
        """Performs a single optimization step.
        Args:
            fisher_info_matrix
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            ngd(params_with_grad,
                d_p_list,
                fisher_info_matrix,
                weight_decay=group['weight_decay'],
                lr=group['lr'],
                )

        return loss

def ngd(params: List[Tensor],
        d_p_list: List[Tensor],
        fisher_info_matrix: FisherInfoMat,
        weight_decay: float,
        lr: float,
        ):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    func = _single_tensor_ngd

    func(params,
        d_p_list,
        fisher_info_matrix=fisher_info_matrix,
        weight_decay=weight_decay,
        lr=lr,
        )

def _single_tensor_ngd(params: List[Tensor],
        d_p_list: List[Tensor],
        fisher_info_matrix: FisherInfoMat,
        weight_decay: float,
        lr: float,
    ):

    for i, param in enumerate(params):
        d_p = d_p_list[i] 

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        d_p = fisher_info_matrix.fvp(
                d_p, 
                use_inverse=True
            )
        param.add_(d_p, alpha=-lr)
