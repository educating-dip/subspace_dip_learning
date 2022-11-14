import torch
import numpy as np

from torch import Tensor
from functorch import vmap, jacrev
from torch.utils.data import DataLoader

class FisherInfoMat:

    def __init__(self, 
        subspace_dip,
        valset: DataLoader, 
        batch_size: int = 1, 
        damping_factor: float = 1e-3
        ):

        self.subspace_dip = subspace_dip
        self.num_subspace_params = len(self.subspace_dip.subspace.parameters_vec)
        self.batch_size = batch_size
        self.num_inputs_valset = len(valset)
        self.matrix = self.init_fisher_info_matrix(
            valset=valset, damping_factor=damping_factor
        )
    
    def fvp(self, 
        v: Tensor,
        use_inverse: bool = False
        ) -> Tensor:

        return self.matrix @ v if not use_inverse else torch.linalg.solve(
                self.matrix, v
            )
    
    def _add_damping(self,  
            matrix: Tensor, 
            damping_factor: float = 1e-3
        ) -> Tensor:

        matrix[np.diag_indices(matrix.shape[0])] += damping_factor
        return matrix

    def compute_fisher_info_matrix(self, 
            valset: DataLoader, 
            damping_factor: float
        ) -> Tensor:
        
        def _fnet_single(params: Tensor, x):
            out = self.subspace_dip.forward(
                    parameters_vec=params, 
                    input=x.unsqueeze(0)
                ).squeeze(0)
            return out

        with torch.no_grad():
            per_inputs_jac_list = []
            for _, _, fbp in valset:
                jac = vmap(
                    jacrev(_fnet_single), (None, 0))(
                        self.subspace_dip.subspace.parameters_vec, fbp
                    )                
                jac = torch.cat([j.flatten() for j in jac]).view(
                        self.batch_size,
                        -1, self.num_subspace_params
                    ) # the inferred dim is im_shape: nn_model_output
                per_inputs_jac_list.append(jac)
            per_inputs_jac = torch.cat(per_inputs_jac_list)
            fisher_info_mat = torch.mean(
                torch.einsum(
                    'Nop,Noc->Npc',
                    per_inputs_jac, 
                    per_inputs_jac), 
                        dim=0) / len(valset)            
            return self._add_damping(
                matrix=fisher_info_mat,
                damping_factor=damping_factor
            ) 

    def init_fisher_info_matrix(self, valset: DataLoader, damping_factor: float = 1e-3) -> Tensor:

        return self.compute_fisher_info_matrix(
                valset=valset, 
                damping_factor=damping_factor
            )

    def update(self, 
        tuneset: DataLoader, 
        mixing_factor: float = 0.5,
        damping_factor: float = 1e-3
        ) -> None:
        
        den = (self.num_inputs_valset + 1)
        self.matrix =  self.num_inputs_valset / den * self.matrix + mixing_factor / den * self.compute_fisher_info_matrix(
                valset=tuneset, 
                damping_factor=damping_factor
            )
        
