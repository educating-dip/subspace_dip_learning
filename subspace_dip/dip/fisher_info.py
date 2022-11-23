from typing import Tuple, Optional

import numpy as np
import torch
import numpy as np
from functools import partial

from torch import Tensor
from functorch import vmap, jacrev, vjp, jvp
from torch.utils.data import DataLoader


def get_random_unit_probes(num_random_vecs, shape):

    new_shape = (num_random_vecs, 1, 1, np.prod(shape))
    v = torch.zeros(*new_shape)
    randinds = torch.randperm(np.prod(shape))[:num_random_vecs]
    v[range(len(randinds)), :, :, randinds] = 1.
    v = v.reshape(num_random_vecs, 1, 1, *shape)
    
    return v

class FisherInfo:

    def __init__(self, 
            subspace_dip,
            valset: DataLoader,
            num_random_vecs: int = 10,
            initial_damping: float = 1e-3,
            mode: str = 'full'
        ):

        self.subspace_dip = subspace_dip
        self.num_inputs_valset = len(valset)
        self.matrix = self.init_fisher_info(
            valset=valset, mode=mode, num_random_vecs=num_random_vecs,
        )
        self._damping = initial_damping

    @property
    def shape(self,
            ) -> Tuple[int,int]:

        size = self.matrix.shape[0]
        return (size, size)

    @property
    def damping(self):
        return self._damping

    @damping.setter
    def damping(self, value):
        self._damping = value
    
    def ema_cvp(self, 
            v: Tensor,
            use_inverse: bool = False,
            include_damping: bool = True,
            include_Tikhonov_regularization: bool = True, 
            weight_decay: float = 0.
        ) -> Tensor:
        
        matrix = self._add_damping(matrix=self.matrix,
            weight_decay=weight_decay if include_Tikhonov_regularization else 0., 
            ) if include_damping else self.matrix

        return matrix @ v if not use_inverse else torch.linalg.solve(matrix, v)

    def exact_cvp(self,             
            v: Tensor,
            use_inverse: bool = False,
            include_damping: bool = True,
            include_Tikhonov_regularization: bool = True, 
            weight_decay: float = 0.,
            use_forward_op: bool = True) -> Tensor:

            if use_inverse: 
                raise NotImplementedError
            
            _fnet_single = partial(self._fnet_single,
                use_forward_op=use_forward_op
                )
            
            _, jvp_ = jvp(_fnet_single,
                    (self.subspace_dip.subspace.parameters_vec,), (v,)
                )
            
            _, _vjp_fn = vjp(_fnet_single,
                        self.subspace_dip.subspace.parameters_vec
                    )
            out = _vjp_fn(jvp_)[0]

            out += v * self.damping if include_damping else 0.
            out += v * weight_decay if include_Tikhonov_regularization else 0.

            return out

    def _add_damping(self,  
            matrix: Tensor,
            weight_decay: float = 0.,  
        ) -> Tensor:

        matrix[np.diag_indices(matrix.shape[0])] += self.damping + weight_decay
        return matrix

    def _fnet_single(self,
            parameters_vec: Tensor,
            input: Optional[Tensor] = None,
            use_forward_op: bool = True
        ) -> Tensor:

        out = self.subspace_dip.forward(
                parameters_vec=parameters_vec,
                input=input,
                **{'use_forward_op':use_forward_op}
            )
        return out

    def assemble_fisher_info(self, 
            valset: DataLoader,
            use_forward_op: bool = True
        ) -> Tensor:
    
        _fnet_single = partial(self._fnet_single,
                use_forward_op=use_forward_op
            )
        with torch.no_grad():
            per_inputs_jac_list = []
            for _, _, fbp in valset:
                jac = vmap(
                    jacrev(_fnet_single), in_dims=(None, 0))(
                        self.subspace_dip.subspace.parameters_vec, fbp.unsqueeze(dim=1)
                    )
                jac = jac.view(
                        fbp.shape[0], #batch_size,
                        -1, self.subspace_dip.subspace.num_subspace_params
                    ) # the inferred dim is im_shape: nn_model_output
                per_inputs_jac_list.append(jac)

            per_inputs_jac = torch.cat(per_inputs_jac_list)
            matrix = torch.mean(
                torch.einsum(
                    'Nop,Noc->Npc',
                    per_inputs_jac, 
                    per_inputs_jac),
                        dim=0)

            return matrix

    def init_fisher_info(self,
            valset: DataLoader,
            num_random_vecs: int = 100, 
            use_forward_op: bool = True,
            mode: str = 'full'
        ) -> Tensor:
        
        if mode == 'full':
            matrix = self.assemble_fisher_info(
                valset=valset,
                use_forward_op=use_forward_op
            )
        elif mode == 'vjp_rank_one':
            matrix = self.random_assemble_fisher_info(                
                valset=valset,
                num_random_vecs=num_random_vecs,
                use_forward_op=use_forward_op
                )
        else: 
            raise NotImplementedError

        return matrix

    def random_assemble_fisher_info(self,
        valset: DataLoader,
        num_random_vecs: int = 10,
        use_forward_op: bool = True
        ) -> Tensor:
        
        if not use_forward_op: 
            shape = self.subspace_dip.ray_trafo.im_shape
        else: 
            shape = self.subspace_dip.ray_trafo.obs_shape

        with torch.no_grad():
            per_inputs_fisher_list = []
            for _, _, fbp in valset:
                
                v = torch.randn((num_random_vecs, 1, 1, *shape), device=self.subspace_dip.device)
                
                _fnet_single = partial(self._fnet_single,
                    input=fbp,
                    use_forward_op=use_forward_op
                    )
                _, _vjp_fn = vjp(_fnet_single,
                        self.subspace_dip.subspace.parameters_vec
                    )
                
                def _single_vjp(v):
                    return _vjp_fn(v)[0]
                
                vJp = vmap(_single_vjp, in_dims=0)(v)
                matrix = torch.einsum('Np,Nc->pc', vJp, vJp) / num_random_vecs
                per_inputs_fisher_list.append(matrix)

            per_inputs_fisher = torch.stack(per_inputs_fisher_list)
            matrix = torch.mean(per_inputs_fisher, dim=0)

        return matrix

    def update(self,
            tuneset: DataLoader,
            curvature_ema: float = 0.95,
            num_random_vecs: int = 10,
            use_forward_op: bool = True,
            mode: str = 'full'
        ) -> None:

        if mode == 'full':
            update = self.assemble_fisher_info(
                valset=tuneset, 
                use_forward_op=use_forward_op
            )
        elif mode == 'vjp_rank_one':
            update = self.random_assemble_fisher_info(
                valset=tuneset,
                use_forward_op=use_forward_op,
                num_random_vecs=num_random_vecs
            )
        else:
            raise NotImplementedError

        matrix = curvature_ema * self.matrix + (1. - curvature_ema) * update
        self.matrix = matrix