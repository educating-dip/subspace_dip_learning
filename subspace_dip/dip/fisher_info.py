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
            num_random_vecs: int = 10, 
            damping_factor: float = 1e-3, 
            mode: str = 'full'
        ):

        self.subspace_dip = subspace_dip
        self.num_inputs_valset = len(valset)
        self.matrix = self.init_fisher_info(
            valset=valset, mode=mode, num_random_vecs=num_random_vecs,
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
            it: int,
            damping_factor: float = 1e-3, 
            mixing_factor: float = 0.1, 
            mode: str = 'fixed'
        ) -> Tensor:

        if mode == 'fixed': 
            fct = damping_factor
        elif mode == 'adaptive': 
            raise NotImplementedError
        else:
            raise NotImplementedError
            # fct = (1 - mixing_factor + mixing_factor**it) * damping_factor
        
        matrix[np.diag_indices(matrix.shape[0])] += fct
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
        num_random_vecs: int = 100,
        use_forward_op: bool = True
        ) -> Tensor:
        
        if not use_forward_op: 
            shape = self.subspace_dip.ray_trafo.im_shape
        else: 
            shape = self.subspace_dip.ray_trafo.obs_shape

        with torch.no_grad():
            per_inputs_fisher_list = []
            for _, _, fbp in valset:

                v = torch.randn(
                    (num_random_vecs, 1, 1, *shape)
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
                matrix = torch.einsum('Np,Nc->pc', vJp, vJp) / num_random_vecs
                per_inputs_fisher_list.append(matrix)

            per_inputs_fisher = torch.stack(per_inputs_fisher_list)
            matrix = torch.mean(per_inputs_fisher, dim=0)

        return matrix


    def update(self,
            tuneset: DataLoader,
            it: int, 
            mixing_factor: float = 0.5,
            damping_factor: float = 1e-3,
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

        matrix = mixing_factor * self.matrix + (1. - mixing_factor) * update
        self.matrix = self._add_damping(
            matrix=matrix,
            it=it,
            damping_factor=damping_factor, 
            mixing_factor=mixing_factor
            )