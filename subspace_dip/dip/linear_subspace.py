from typing import Optional, List, Dict

import os
import socket
import datetime
import numpy as np
import torch
import torch as Tensor
import torch.nn as nn
import tensorly as tl
tl.set_backend('pytorch')
import tensorboardX
from math import ceil 
from tqdm import tqdm
from torch.utils.data import DataLoader

from .utils import gramschmidt
from subspace_dip.utils import get_original_cwd
from subspace_dip.utils import PSNR, tv_loss
from subspace_dip.data import BaseRayTrafo

class LinearSubspace(nn.Module):
    def __init__(self, 
        parameters_samples_list: Optional[List] = None, 
        use_random_init: bool = True,
        subspace_dim: Optional[int] = None,
        num_random_projs: Optional[int] = None,
        use_approx: bool = False,
        remove_first_n_samples: Optional[int] = None,
        load_ortho_basis_path: Optional[str] = None,
        params_space_retain_ftc: Optional[float] = None,
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
                use_approx=use_approx, 
                remove_first_n_samples=remove_first_n_samples
                )
        else: 
            self.load_ortho_basis(ortho_basis_path=load_ortho_basis_path)

        self.params_space_retain_ftc = params_space_retain_ftc
        self.is_trimmed = False
        if self.params_space_retain_ftc is not None:
            self.is_trimmed = True
            self._trimming_params_in_subspace()

        self.init_parameters(use_random_init=use_random_init)
        self.num_subspace_params = len(self.parameters_vec)
    
    def _trimming_params_in_subspace(self):

        lev_score = self.ortho_basis.pow(2).sum(dim=1) # sum over subspace_dim
        num_params_to_be_retained = ceil(self.params_space_retain_ftc*self.ortho_basis.shape[0])
        _, indices = torch.topk(lev_score, k=num_params_to_be_retained)
        # (num_params, subspace_dim) -> (num_params_to_be_retained, subspace_dim)
        self.ortho_basis = self.ortho_basis[indices, :] # this frees memory due to indices being array 
        self.indices = indices.tolist()

    def init_parameters(self,
        use_random_init: bool = True,
        ) -> None:
    
        init_parameters = torch.zeros(
            self.ortho_basis.shape[-1],
            requires_grad=True,
            device=self.device
            )
        if use_random_init: 
            init_parameters = torch.randn_like(
                init_parameters, 
                requires_grad=True
            )
            init_parameters = init_parameters / init_parameters.pow(2).sum()
        self.parameters_vec = nn.Parameter(init_parameters)
        
    def save_ortho_basis(self, 
        name: str = 'ortho_basis',
        ortho_basis_path: str = './'
        ):

        path = ortho_basis_path if ortho_basis_path.endswith('.pt') else ortho_basis_path + name + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'ortho_basis': self.ortho_basis, 'singular_values' : self.singular_values}, path)

    def load_ortho_basis(self, 
        ortho_basis_path: str, 
        ):

        path = os.path.join(get_original_cwd(), 
            ortho_basis_path if ortho_basis_path.endswith('.pt') \
                else ortho_basis_path + '.pt')
        self.ortho_basis = torch.load(path, map_location=self.device)['ortho_basis']
        self.singular_values = torch.load(path, map_location=self.device)['singular_values']

    def extract_ortho_basis_subspace(self,
        subspace_dim: Optional[int] = None,
        num_random_projs: Optional[int] = None,
        return_singular_values: Optional[bool] = True,
        device = None,
        use_cpu: bool = True, 
        use_approx: bool = False, 
        remove_first_n_samples: Optional[int] = None
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

        if not use_approx:
            # https://github.com/tensorly/tensorly/blob/15d9647e08dee10c990fe2731a1b92db6428bad9/tensorly/contrib/sparse/backend/numpy_backend.py
            ortho_bases, singular_values, _  = tl.partial_svd(
                params_mat,
                n_eigenvecs=subspace_dim
                )
        else:
            # https://docs.dask.org/en/stable/generated/dask.array.linalg.svd_compressed.html
            # params_mat_da = da.from_array(
            #         params_mat.numpy() if not params_mat.is_cuda else params_mat.cpu().numpy()
            #     )
            # out = da.linalg.svd_compressed(params_mat_da, k=subspace_dim)
            # ortho_bases = torch.from_numpy(out[0].persist().compute())
            # singular_values = torch.from_numpy(out[1].persist().compute())
            if remove_first_n_samples is None: 
                remove_first_n_samples = 0

            ortho_bases, singular_values, _ = torch.svd_lowrank(
                params_mat[:, remove_first_n_samples:], q=subspace_dim) # analogous as commented above

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
        return ortho_bases.to(device=device) if not return_singular_values else (
            ortho_bases.to(device=device), singular_values.to(device=device)
        )

    def set_parameters_on_valset(self,
        subspace_dip,
        ray_trafo: BaseRayTrafo,
        valset: DataLoader, 
        optim_kwargs: Dict,
        ):

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'finetune_paramerters_on_testset'
        logdir = os.path.join(
            optim_kwargs['log_path'],
            current_time + '_' + socket.gethostname() + '_' + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

        if optim_kwargs['torch_manual_seed']:
            torch.random.manual_seed(optim_kwargs['torch_manual_seed'])


        subspace_dip.nn_model.train()

        criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [self.parameters_vec],
            lr=optim_kwargs['optim']['lr'],
            weight_decay=optim_kwargs['optim']['weight_decay']
            )
                
        running_psnr = 0.0
        running_loss = 0.0
        running_size = 0
        i = 0
        for epoch in range(optim_kwargs['epochs']):
            # Each epoch has a training and validation phase
                with tqdm(valset,
                        desc='epoch {:d}'.format(epoch + 1) ) as pbar:
                    for observation, gt, fbp in pbar:
                        
                        observation = observation.to(self.device)
                        gt = gt.to(self.device)
                        fbp = fbp.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        outputs = subspace_dip(
                                input=fbp, 
                                apply_forward_op=False
                            )
                        loss = criterion(ray_trafo(outputs), observation) 
                        loss = loss + optim_kwargs['optim']['gamma']*tv_loss(outputs)

                        # backward
                        loss.backward()
                        self.optimizer.step()

                        for i in range(outputs.shape[0]):
                            gt_ = gt[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, gt_, data_range=1)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})

                        self.writer.add_scalar('loss', running_loss/running_size, i)
                        self.writer.add_scalar('psnr', running_psnr/running_size, i)
                        i += 1