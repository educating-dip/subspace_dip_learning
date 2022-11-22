from typing import Dict, List, Optional, Tuple
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
                        weight_decay=weight_decay, differentiable=differentiable)
        self.mem_state = {'delta': None, 'rho': [], 'damping': []}

        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, 
            fisher_info: FisherInfo,
            use_adaptive_damping: bool = False,
            it: int = None,
            closure=None, 
            loss=None,
        ):
        
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

            memory = ngd(params_with_grad,
                d_p_list,
                fisher_info,
                weight_decay=group['weight_decay'],
                lr=group['lr'],
                use_adaptive_damping=use_adaptive_damping,
                mem_state=self.mem_state,
                it=it,
                closure=closure, 
                loss=loss
                )

            if use_adaptive_damping: 
                self.mem_state['delta'] = memory[0]
                self.mem_state['rho'].append(memory[1])
                self.mem_state['damping'].append(memory[2])


def ngd(params: List[Tensor],
        d_p_list: List[Tensor],
        fisher_info: FisherInfo,
        weight_decay: float,
        lr: float,
        use_adaptive_damping: bool = False,
        mem_state: Optional[Dict[str, List[Tensor]]] = {},
        it: Optional[int] = None, 
        closure=None,
        loss=None
        ) -> Tuple[Tensor, float, float]:


    r"""Functional API that performs NGD algorithm computation.
    """

    func = _single_tensor_ngd

    memory = func(params,
        d_p_list,
        fisher_info=fisher_info,
        weight_decay=weight_decay,
        lr=lr,
        use_adaptive_damping=use_adaptive_damping,
        mem_state=mem_state,
        it=it,
        closure=closure,
        loss=loss
        )

    if use_adaptive_damping: return memory

def _compute_adaptive_damping_via_reduction_ratio(
        fisher_info: FisherInfo, 
        closure: callable, 
        loss: Tensor, 
        param: Tensor,
        delta: Tensor,
        weight_decay: float,
        T: float, 
        decay_fct: float,
    ) -> Tuple[float,float]:
    
    red_pred_by_quad_model = .5*delta @ fisher_info.fvp(
            delta, 
            weight_decay=weight_decay, 
            use_inverse=False
        ) + param.grad @ delta
    
    n_params = param + delta
    n_loss = closure(parameters_vec=n_params)[0]
    red_in_obj = n_loss - loss # missing weight decay 
    rho = red_in_obj / red_pred_by_quad_model
    damping_fct = fisher_info.damping_fct

    if rho < 0.25:
        damping_fct = decay_fct**(-T)*damping_fct
    elif rho > 0.75:
        damping_fct = decay_fct**(T)*damping_fct

    return damping_fct, rho.item()

def _single_tensor_ngd(params: List[Tensor],
        d_p_list: List[Tensor],
        fisher_info: FisherInfo,
        mem_state: Optional[Dict[str, List[Tensor]]] = {},
        weight_decay: float = 0.,
        lr: float = 1e-3, 
        use_adaptive_damping: bool = False,
        closure = None,
        loss = None, 
        it: Optional[int] = None,
        T: int = 5
    )-> Tuple[Tensor, float, float]:

    for i, param in enumerate(params):
        d_p = d_p_list[i] 

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        
        damping, rho = None, None
        if use_adaptive_damping and ( (it + 1) % T == 0):
            
            delta = mem_state['delta']
            adaptive_damping_kwargs = {
                'weight_decay': weight_decay,
                'T': T, 
                'decay_fct': (19/20)
                }
            damping, rho = _compute_adaptive_damping_via_reduction_ratio(
                fisher_info=fisher_info,
                closure=closure,
                param=param,
                loss=loss, 
                delta=delta,
                **adaptive_damping_kwargs
            )
            fisher_info.damping_fct = damping

        d_p = fisher_info.fvp(
                d_p,
                weight_decay=weight_decay,
                use_inverse=True
            )
        param.add_(d_p, alpha=-lr)

        if use_adaptive_damping:
            return -lr*d_p, rho, damping
    
