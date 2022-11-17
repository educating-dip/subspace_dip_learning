from typing import Tuple

import numpy as np
import torch
import numpy as np
from functools import partial

from torch import Tensor
from functorch import vmap, jacrev, vjp
from torch.utils.data import DataLoader

class FisherInfo:

    def __init__(self, 
            subspace_dip,
            valset: DataLoader, 
            batch_size: int = 1, 
            damping_factor: float = 1e-3
        ):

        self.subspace_dip = subspace_dip
        self.batch_size = batch_size
        self.num_inputs_valset = len(valset)
        self.matrix = self.init_fisher_info_matrix(
            valset=valset, damping_factor=damping_factor
        )

    @property
    def shape(self, 
            ) -> Tuple[int,int]:

        size = self.matrix.shape[0]
        return (size, size)
 
    def Fvp(self, 
            v: Tensor,
            use_inverse: bool = False
        ) -> Tensor:

        return self.matrix @ v if not use_inverse else torch.linalg.solve(self.matrix, v)
    
    def _add_damping(self,  
            matrix: Tensor, 
            damping_factor: float = 1e-3
        ) -> Tensor:

        matrix[np.diag_indices(matrix.shape[0])] += damping_factor
        return matrix

    def _fnet_single(self,
            parameters_vec: Tensor,
            input: Tensor,
            use_forward_op: bool = True
        ) -> Tensor:

        out = self.subspace_dip.forward(
                parameters_vec=parameters_vec,
                input=input,
                **{'use_forward_op':use_forward_op}
            )
        return out

    def assemble_fisher_info_matrix(self, 
            valset: DataLoader,
            damping_factor: float,
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
                        self.batch_size,
                        -1, self.subspace_dip.subspace.num_subspace_params
                    ) # the inferred dim is im_shape: nn_model_output
                per_inputs_jac_list.append(jac)
            per_inputs_jac = torch.cat(per_inputs_jac_list)
            fisher_info_mat = torch.mean(
                torch.einsum(
                    'Nop,Noc->Npc',
                    per_inputs_jac, 
                    per_inputs_jac),
                        dim=0)
     
            return self._add_damping(
                matrix=fisher_info_mat,
                damping_factor=damping_factor
            ) 

    def init_fisher_info_matrix(self,
            valset: DataLoader,
            damping_factor: float = 1e-3,
            use_forward_op: bool = True, 
        ) -> Tensor:

        return self.assemble_fisher_info_matrix(
                valset=valset, 
                damping_factor=damping_factor,
                use_forward_op=use_forward_op
            )

    def randm_rank_one_update(self,
        tuneset: DataLoader,
        batch_size: int = 50,
        use_forward_op: bool = True
        ) -> Tensor:
        
        if not use_forward_op: 
            shape = self.subspace_dip.ray_trafo.im_shape
        else: 
            shape = self.subspace_dip.ray_trafo.obs_shape

        _, _, fbp = next(iter(tuneset))
        with torch.no_grad():
            v = torch.randn(
                (batch_size, 1, 1, *shape)
                    ).to(device=self.subspace_dip.device)

            v /= v.norm(dim=0, keepdim=True)

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
            update = torch.einsum('Np,Nc->pc', vJp, vJp)
        
        return update

    def update(self,
            tuneset: DataLoader,
            mixing_factor: float = 0.5,
            damping_factor: float = 1e-3,
            use_forward_op: bool = True,
            mode: str = 'full'
        ) -> None:
        
        if mode == 'full': 
            update = self.assemble_fisher_info_matrix(
                valset=tuneset, 
                damping_factor=damping_factor,
                use_forward_op=use_forward_op
            )
        elif mode == 'jvp_rank_one':
            update = self.randm_rank_one_update(
                tuneset=tuneset, 
            )
        
        else: 
            raise NotImplementedError

        self.matrix = mixing_factor * self.matrix + (1. - mixing_factor) * update
