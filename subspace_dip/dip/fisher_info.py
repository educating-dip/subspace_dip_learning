from typing import Tuple, Optional

import numpy as np
import torch
import numpy as np
from functools import partial

from torch import Tensor
from functorch import vmap, jacrev, vjp, jvp
from torch.utils.data import DataLoader
from .utils import generate_random_unit_probes

class Damping:

    def __init__(self, init_damping: float = 1e2):
        self._damping = init_damping
    
    @property
    def damping(self, ): 
        return self._damping

    @damping.setter
    def damping(self, value): 
        self._damping = value 

    def add_damping(self,  
            matrix: Tensor,
            include_Tikhonov_regularization: bool = True,
            weight_decay: float = 0.
        ) -> Tensor:

        matrix[np.diag_indices(matrix.shape[0])] += self.damping
        if include_Tikhonov_regularization: 
            matrix[np.diag_indices(matrix.shape[0])] += weight_decay

        return matrix

class FisherInfo:

    def __init__(self, 
            subspace_dip,
            init_damping: float = 1e-3,
            use_uniform_vjp_smpl_probs: bool = None
        ):

        self.subspace_dip = subspace_dip
        self.matrix = torch.eye(self.subspace_dip.subspace.num_subspace_params, 
            device=self.subspace_dip.device
        )
        self.curvature_damping = Damping(init_damping=init_damping)
        # first level proxy for AJU
        row_norm = self.subspace_dip.ray_trafo.matrix.norm(dim=1)
        self._vjp_smpl_probs = row_norm.cpu().numpy() / torch.sum(row_norm).cpu().numpy() if not use_uniform_vjp_smpl_probs else None 

    @property
    def shape(self,
            ) -> Tuple[int,int]:
        size = self.matrix.shape[0]
        return (size, size)
    
    def ema_cvp(self, 
            v: Tensor,
            use_inverse: bool = False,
            use_square_root: bool = False, 
            include_damping: bool = False, # include λ
            include_Tikhonov_regularization: bool = False, # include η
            weight_decay: float = 0.
        ) -> Tensor:

        matrix = self.matrix.clone() if (
            include_damping or include_Tikhonov_regularization
                ) else self.matrix
    
        if use_square_root: 
            chol = torch.linalg.cholesky(matrix)
            return chol @ v
        else:
            if include_damping:
                matrix = self.curvature_damping.add_damping(matrix=matrix, 
                    include_Tikhonov_regularization=include_Tikhonov_regularization
                ) # add λ
            return matrix @ v if not use_inverse else torch.linalg.solve(matrix, v)

    def exact_cvp(self,             
            v: Tensor,
            use_forward_op: bool = True,
            use_square_root: bool = False
            ) -> Tensor:

            _fnet_single = partial(self._fnet_single,
                use_forward_op=use_forward_op
                )
            _, jvp_ = jvp(_fnet_single, 
                    (self.subspace_dip.subspace.parameters_vec,), (v,)
                ) # jvp_ = v @ J_cT = v @ (UT JT AT)
            if not use_square_root: 
                _, _vjp_fn = vjp(_fnet_single,
                            self.subspace_dip.subspace.parameters_vec
                        )
                Fv = _vjp_fn(jvp_)[0] # Fv = jvp_ @ J_c = jvp_ @ (A J U)
            else: 
                Fv = jvp_
            return Fv

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
            dataset: Optional[DataLoader] = None,
            use_forward_op: bool = True
        ) -> Tensor:

        def _per_input_full_update(fbp: Optional[Tensor] = None): 
                
            _fnet_single = partial(self._fnet_single,
                use_forward_op=use_forward_op
            )

            if fbp is None:
                jac = jacrev(
                        _fnet_single
                    )(self.subspace_dip.subspace.parameters_vec)
            else:
                jac = vmap(jacrev(_fnet_single), in_dims=(None, 0))(
                    self.subspace_dip.subspace.parameters_vec, fbp.unsqueeze(dim=1)
                )
    
            jac = jac.view(
                fbp.shape[0] if fbp is not None else 1, #batch_size,
                -1, self.subspace_dip.subspace.num_subspace_params
            ) # the inferred dim is im_shape: nn_model_output
            return jac
        
        with torch.no_grad():
            per_inputs_jac_list = []
            if dataset is not None:
                for _, _, fbp in dataset:
                    jac = _per_input_full_update(fbp=fbp)
                    per_inputs_jac_list.append(jac)
            else:
                jac = _per_input_full_update(fbp=None)
                per_inputs_jac_list.append(jac)

            per_inputs_jac = torch.cat(per_inputs_jac_list)
            matrix = torch.mean(
                torch.einsum(
                    'Nop,Noc->Npc',
                    per_inputs_jac, 
                    per_inputs_jac),
                        dim=0)

            return matrix

    def initialise_fisher_info(self,
            dataset: Optional[DataLoader] = None,
            num_random_vecs: int = 100, 
            use_forward_op: bool = True,
            mode: str = 'full'
        ) -> Tensor:
        
        if mode == 'full':
            matrix = self.assemble_fisher_info(
                dataset=dataset,
                use_forward_op=use_forward_op
            )
        elif mode == 'vjp_rank_one':
            matrix = self.random_assemble_fisher_info(                
                dataset=dataset,
                num_random_vecs=num_random_vecs,
                use_forward_op=use_forward_op
                )
        else: 
            raise NotImplementedError
        self.matrix = matrix 

    def random_assemble_fisher_info(self,
        dataset: Optional[DataLoader] = None,
        num_random_vecs: int = 10,
        use_forward_op: bool = True
        ) -> Tensor:
        
        if not use_forward_op: 
            shape = self.subspace_dip.ray_trafo.im_shape
        else: 
            shape = self.subspace_dip.ray_trafo.obs_shape

        def _per_input_rank_one_update(fbp: Optional[Tensor] = None):

            v = generate_random_unit_probes(
                num_random_vecs, shape, p=self._vjp_smpl_probs
            ).to(device=self.subspace_dip.device)
            
            _fnet_single = partial(self._fnet_single,
                input=fbp,
                use_forward_op=use_forward_op
                )
            _, _vjp_fn = vjp(_fnet_single,
                    self.subspace_dip.subspace.parameters_vec
                )
            
            def _single_vjp(v):
                return _vjp_fn(v)[0]
            
            vJp = vmap(_single_vjp, in_dims=0)(v) # vJp = v.T @ (AJU)
            matrix = torch.einsum('Np,Nc->pc', vJp, vJp) * np.prod(shape) / num_random_vecs

            return matrix

        with torch.no_grad():
            per_inputs_fisher_list = []
            if dataset is not None: 
                for _, _, fbp in dataset:
                    matrix = _per_input_rank_one_update(fbp=fbp)
                    per_inputs_fisher_list.append(matrix)
            else:
                matrix = _per_input_rank_one_update(fbp=None)
                per_inputs_fisher_list.append(matrix)

            per_inputs_fisher = torch.stack(per_inputs_fisher_list)
            matrix = torch.mean(per_inputs_fisher, dim=0)

        return matrix

    def update(self,
            dataset: Optional[DataLoader] = None,
            curvature_ema: float = 0.95,
            num_random_vecs: int = 10,
            use_forward_op: bool = True,
            mode: str = 'full'
        ) -> None:

        if mode == 'full':
            update = self.assemble_fisher_info(
                dataset=dataset, 
                use_forward_op=use_forward_op
            )
        elif mode == 'vjp_rank_one':
            update = self.random_assemble_fisher_info(
                dataset=dataset,
                use_forward_op=use_forward_op,
                num_random_vecs=num_random_vecs
            )
        else:
            raise NotImplementedError
        matrix = curvature_ema * self.matrix + (1. - curvature_ema) * update
        self.matrix = matrix