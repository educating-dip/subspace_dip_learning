from typing import Tuple, Optional

import numpy as np
import torch
import numpy as np
from functools import partial

from torch import Tensor
from functorch import vmap, jacrev, vjp, jvp, jacfwd
from torch.utils.data import DataLoader

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

class SamplingProbes:
    def __init__(self, prxy_mat=None, mode='row_norm', device=None):

        self.prxy_mat = prxy_mat
        self.mode = mode
        self.device = device
        
        if self.mode == 'row_norm':
            assert self.prxy_mat is not None
            assert  self.prxy_mat.ndim == 2

            un_ps = torch.linalg.norm(self.prxy_mat, dim=1, ord=2).pow(2)
            const = un_ps.sum()
            self.ps = un_ps/const

        elif self.mode == 'gauss': 
            pass
        else: 
            raise NotImplementedError

    def sample_probes(self, num_random_vecs: int, shape: Tuple[int, int]) -> Tensor:

        if self.mode == 'row_norm': 
            func = self._scaled_unit_probes

        elif self.mode == 'gauss':
            def _gauss_probes(num_random_vecs, shape):
                return torch.randn(
                    (num_random_vecs, 1, 1, *shape), 
                        device=self.device)
            func = _gauss_probes
        else:
            raise NotImplementedError

        return func(num_random_vecs=num_random_vecs, shape=shape)

    def _scaled_unit_probes(self, num_random_vecs, shape):

        new_shape = (num_random_vecs, 1, 1, np.prod(shape))
        v = torch.zeros(*new_shape, device=self.device)
        rand_inds = np.random.choice(
                np.prod(shape), 
                size=num_random_vecs, 
                replace=True, 
                p=self.ps.cpu().numpy()
            )
        ps_mask = self.ps.expand(num_random_vecs, -1).view(new_shape).pow(-.5)
        v[range(num_random_vecs), :, :, rand_inds] = ps_mask[range(num_random_vecs), :, :, rand_inds]

        return v.reshape(num_random_vecs, 1, 1, *shape)

def _ema_polynomial_scheduler(
    step_cnt: int,
    base_curvature_ema: float = 0.5,
    max_iterations: int = 1000, 
    power: float = .99, 
    max_ema: float = 0.95, 
    increase: bool = True
    ) -> float:

    fct = np.clip(step_cnt / max_iterations, a_max=1, a_min=0)
    return np.clip(base_curvature_ema * (
        (1.1 - fct) ** (- power*increase + power*(not increase) )
            ), a_min=1-max_ema, a_max=max_ema)

class FisherInfo:

    def __init__(self, 
        subspace_dip,
        init_damping: float = 1e-3,
        curvature_ema: float = 0.95,
        sampling_probes_mode: str = 'row_norm'
        ):

        self.subspace_dip = subspace_dip
        self.matrix = torch.eye(self.subspace_dip.subspace.num_subspace_params, 
            device=self.subspace_dip.device
        )
        self.init_matrix = None
        self.curvature_damping = Damping(init_damping=init_damping)
        self._curvature_ema = curvature_ema
        self.probes = SamplingProbes(
                prxy_mat=self.subspace_dip.ray_trafo.matrix, 
                mode=sampling_probes_mode, 
                device=self.subspace_dip.device
            )
    
    @property
    def shape(self, ) -> Tuple[int,int]:
        size = self.matrix.shape[0]
        return (size, size)
    
    @property
    def curvature_ema(self, ) -> float:
        return self._curvature_ema 

    @curvature_ema.setter
    def curvature_ema(self, value) -> None:
        self._curvature_ema = value
    
    def update_curvature_ema(self, step_cnt: int, update_kwargs) -> None:
        self.curvature_ema = _ema_polynomial_scheduler(step_cnt=step_cnt, **update_kwargs)

    def reset_fisher_matrix(self, ) -> None:
        self.matrix = self.init_matrix.clone() if self.init_matrix is not None else torch.eye(
            self.subspace_dip.subspace.num_subspace_params, device=self.subspace_dip.device)

    def approx_fisher_vp(self,
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
            # it returns the upper triangular matrix 
            chol = torch.linalg.cholesky(matrix, upper=True) 
            return chol@v 
        else:
            if include_damping:
                matrix = self.curvature_damping.add_damping(matrix=matrix, 
                    include_Tikhonov_regularization=include_Tikhonov_regularization, 
                    weight_decay=weight_decay
                ) # add λ and η
            return matrix @ v if not use_inverse else torch.linalg.solve(matrix, v)

    def exact_fisher_vp(self,             
        v: Tensor,
        use_forward_op: bool = True,
        use_square_root: bool = False
        ) -> Tensor:

            _fnet_single = partial(self._fnet_single, use_forward_op=use_forward_op)
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

    def exact_fisher_assembly(self, 
        dataset: Optional[DataLoader] = None,
        use_forward_op: bool = True
        ) -> Tensor:

        def _per_input_exact_update(fbp: Optional[Tensor] = None): 
                
            _fnet_single = partial(self._fnet_single,use_forward_op=use_forward_op)
            if fbp is None:
                jac = jacrev(_fnet_single)(self.subspace_dip.subspace.parameters_vec)
            else:
                jac = vmap(jacrev(_fnet_single), in_dims=(None, 0))(
                    self.subspace_dip.subspace.parameters_vec, fbp.unsqueeze(dim=1)
                )
    
            jac = jac.view(fbp.shape[0] if fbp is not None else 1, #batch_size,
                    -1, self.subspace_dip.subspace.num_subspace_params
                ) # the inferred dim is im_shape: nn_model_output
            return jac # (batch_size, nn_model_output, num_subspace_params)
        
        with torch.no_grad():
            per_inputs_jac_list = []
            if dataset is not None:
                for _, _, fbp in dataset:
                    jac = _per_input_exact_update(fbp=fbp)
                    per_inputs_jac_list.append(jac)
            else:
                jac = _per_input_exact_update(fbp=None)
                per_inputs_jac_list.append(jac)

            per_inputs_jac = torch.cat(per_inputs_jac_list)
            # same as (per_inputs_jac.mT @ per_inputs_jac).mean(dim=0)
            matrix = torch.mean(torch.einsum('Nop,Noc->Npc', per_inputs_jac, per_inputs_jac), dim=0)

            return matrix, jac

    def initialise_fisher_info(self,
        dataset: Optional[DataLoader] = None,
        num_random_vecs: int = 100, 
        use_forward_op: bool = True,
        mode: str = 'full'
        ) -> Tensor:
        
        if mode == 'full':
            matrix, _ = self.exact_fisher_assembly(dataset=dataset,
                use_forward_op=use_forward_op
            )
        elif mode == 'vjp_rank_one':
            matrix = self.online_fisher_assembly(dataset=dataset,
                num_random_vecs=num_random_vecs, use_forward_op=use_forward_op
            )
        else: 
            raise NotImplementedError
        self.matrix = matrix
        self.init_matrix = matrix

    def online_fisher_assembly(self,
        dataset: Optional[DataLoader] = None,
        num_random_vecs: int = 10,
        use_forward_op: bool = True
        ) -> Tensor:
        
        if not use_forward_op: 
            shape = self.subspace_dip.ray_trafo.im_shape
        else: 
            shape = self.subspace_dip.ray_trafo.obs_shape

        def _per_input_rank_one_update(fbp: Optional[Tensor] = None):

            v = self.probes.sample_probes(num_random_vecs=num_random_vecs, shape=shape)
            _fnet_single = partial(self._fnet_single,
                input=fbp,
                use_forward_op=use_forward_op
                )
            _, _vjp_fn = vjp(_fnet_single,
                    self.subspace_dip.subspace.parameters_vec
                )

            def _single_vjp(v):
                return _vjp_fn(v)[0]

            vJp = vmap(_single_vjp, in_dims=0)(v) #vJp = v.T@(AJU)
            matrix = torch.einsum('Np,Nc->pc', vJp, vJp) / num_random_vecs
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
        num_random_vecs: Optional[int] = 10,
        use_forward_op: bool = True,
        mode: str = 'full'
        ) -> None:

        if mode == 'full':
            update, _ = self.exact_fisher_assembly(
                dataset=dataset, use_forward_op=use_forward_op
            )
        elif mode == 'vjp_rank_one':
            update = self.online_fisher_assembly(dataset=dataset,
                use_forward_op=use_forward_op, num_random_vecs=num_random_vecs
            )
        else:
            raise NotImplementedError
        matrix = curvature_ema * self.matrix + (1. - curvature_ema) * update
        self.matrix = matrix