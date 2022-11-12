import torch
import numpy as np

from torch import Tensor
from functorch import vmap, jacrev
from torch.utils.data import DataLoader

class FisherInfoMat:

    def __init__(self, 
        subspace_dip,
        valset, 
        batch_size, 
        im_shape,
        ):

        self.subspace_dip = subspace_dip
        self.valset = valset
        self.batch_size = batch_size
        self.im_shape = im_shape
        self._matrix = self.init_fisher_info_matrix()
    
    @property
    def matrix(self, ): 
        return self._matrix
    
    def fvp(self, 
        v: Tensor,
        use_inverse: bool = False
        ) -> Tensor:
        
        return self.matrix @ v if not use_inverse else torch.linalg.solve(
                self.matrix, v
            )
    
    def compute_fisher_info(self, valset: DataLoader) -> Tensor:
        
        def _fnet_single(params, x):
            out = self.subspace_dip.func_model_with_input(
                    params, 
                    x.unsqueeze(0)
                ).squeeze(0)
            return out

        with torch.no_grad():
            per_input_jac_list = []
            for _, _, fbp in valset:
                jac = vmap(
                    jacrev(_fnet_single), (None, 0))(
                        self.subspace_dip.get_func_params(), fbp
                    )                
                jac = torch.cat([j.flatten() for j in jac]).view(
                    self.batch_size, 
                    np.prod(self.im_shape), -1
                )
                per_input_jac_list.append(jac)
            per_input_jac = torch.cat(per_input_jac_list)
            per_input_jac_proj = per_input_jac @ self.subspace_dip.subspace.ortho_basis
            fisher_info_mat = torch.mean(
                torch.einsum(
                    'Nop,Noc->Npc', 
                    per_input_jac_proj, 
                    per_input_jac_proj), 
                        dim=0) / len(self.valset)
            return fisher_info_mat

    def init_fisher_info_matrix(self, ) -> Tensor:

        return self.compute_fisher_info(
                valset=self.valset
            )

    def update(self, 
        tuneset: DataLoader, 
        mixing_factor: float = 0.5
        ) -> None:
        
        den = (len(self.valset) + 1)
        self._matrix = len(self.valset) / den * self._matrix + mixing_factor / den * self.compute_fisher_info(
                valset=tuneset
            )
        
