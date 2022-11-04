from typing import Optional, List

import os
import torch
import torch as Tensor
import torch.nn as nn
import tensorly as tl 
tl.set_backend('pytorch')

from .utils import gramschmidt
from subspace_dip.utils import get_original_cwd

class LinearSubspace(nn.Module):
    def __init__(self, 
        parameters_samples_list: Optional[List] = None, 
        use_random_init: bool = True,
        subspace_dim: Optional[int] = None,
        num_random_projs: Optional[int] = None,
        load_ortho_basis_path: Optional[str] = None,
        device = None
        ) -> None:

        super().__init__()
        
        assert not (load_ortho_basis_path and parameters_samples_list)
        
        self.device = device or torch.device(
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )

        if parameters_samples_list is not None: 
            self.parameters_samples_list = parameters_samples_list
            self.ortho_basis, self.singular_values = self.extract_ortho_basis_subspace(
                subspace_dim=subspace_dim,
                num_random_projs=num_random_projs, 
                )
        else: 
            self.load_ortho_basis(ortho_basis_path=load_ortho_basis_path)

        self._init_parameters(use_random_init=use_random_init)

    def _init_parameters(self, 
        use_random_init: bool = True
        ) -> None:
    
        init_parameters = torch.zeros(
            self.ortho_basis.shape[-1],
            requires_grad=True,
            device=self.device
            )
        if use_random_init: 
            init_parameters = torch.randn_like(init_parameters)
            init_parameters /= init_parameters.pow(2).sum()
        self.parameters_vec = nn.Parameter(init_parameters)

    def save_ortho_basis(self, 
        name: str = 'ortho_basis',
        ortho_basis_path: str = './'
        ):

        path = ortho_basis_path if ortho_basis_path.endswith('.pt') else ortho_basis_path + name + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.ortho_basis, path)

    def load_ortho_basis(self, 
        ortho_basis_path: str, 
        ):

        path = os.path.join(get_original_cwd(), 
            ortho_basis_path if ortho_basis_path.endswith('.pt') \
                else ortho_basis_path + '.pt')
        self.ortho_basis = torch.load(path, map_location=self.device)

    def extract_ortho_basis_subspace(self,
        subspace_dim: Optional[int] = None,
        num_random_projs: Optional[int] = None,
        return_singular_values: Optional[bool] = True,
        device = None, 
        use_cpu: bool = True
        ) -> Tensor:

        def _add_random_projs(
                ortho_bases: Tensor,
                num_random_projs: int
                ) -> Tensor:
            
            randn_projs = torch.randn((ortho_bases.shape[0], num_random_projs))
            return gramschmidt(
                ortho_bases=ortho_bases,
                randn_projs=randn_projs
            )

        subspace_dim = subspace_dim if subspace_dim is not None else len(self.parameters_samples_list)
        params_mat = torch.moveaxis(
            torch.stack(self.parameters_samples_list), (0, 1), (1, 0)
            ) # (num_params, subspace_dim)
        params_mat = params_mat if not use_cpu else params_mat.cpu()
        ortho_bases, singular_values, _  = tl.partial_svd(
            params_mat, 
            n_eigenvecs=subspace_dim
            )
        
        if num_random_projs is not None: 
            ortho_bases = _add_random_projs(
                ortho_bases=ortho_bases,
                num_random_projs=num_random_projs
            )
        
        """
        Returns
        -------
        ortho_bases : Tensor Size. (num_params, subspace_dim or subspace_dim+num_random_projs)
        """
        return ortho_bases.detach().to(device=device) if not return_singular_values else (
            ortho_bases.detach().to(device=device), singular_values.detach().to(device=device)
        )
