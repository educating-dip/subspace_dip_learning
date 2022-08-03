"""
Provides :class:`SubspaceConstructor`.
"""
from typing import Dict, Optional, List
import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
import tensorly as tl 
tl.set_backend('pytorch')
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from math import ceil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from subspace_dip.utils import PSNR, normalize, get_original_cwd
from subspace_dip.data import get_ellipses_dataset
from subspace_dip.utils import get_params_from_nn_module
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo


class SubspaceConstructor:

    """
    Wrapper for constructing a low-dimensional subspace of the NN optimisation trajectory.
    """

    def __init__(self, 
            model : nn.Module,
            exclude_norm_layers : bool = False, 
            include_bias : bool = True, 
            device = None, 
            ):
        
        self.model = model
        self.exclude_norm_layers = exclude_norm_layers
        self.include_bias = include_bias
        self.device = device or torch.device(
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.params_traj_samples = []

    def add_params_traj_samples(self, ) -> List[Tensor]:
        
        self.params_traj_samples.append(
            get_params_from_nn_module(
                self.model,
                exclude_norm_layers=self.exclude_norm_layers,
                include_bias=self.include_bias
            )
        )
    
    def save_params_traj_samples(self, 
        name: str = 'params_traj_samples',
        path: str = './'
    ):

        path = path if path.endswith('.pt') else path + name + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.params_traj_samples, path)

         
    def load_params_traj_samples(self, 
        path_to_params_traj_samples: str, 
        device = None
        ):
        
        path = os.path.join(get_original_cwd(), 
            path_to_params_traj_samples if path_to_params_traj_samples.endswith('.pt') \
                else path_to_params_traj_samples + '.pt')
        self.params_traj_samples.extend(
            torch.load(path, map_location=device)
        )
    
    @classmethod
    def compute_bases_subspace(cls,
            params_traj_samples : List[Tensor], 
            subspace_dim : Optional[int] = None,
            num_rand_projs: Optional[int] = None,
            return_singular_values : bool = False,
            device = None, 
            use_cpu : bool = True
        ) -> Tensor:

        def _add_random_projs(
                bases: Tensor,
                num_rand_projs: int
                ) -> Tensor:
            
            randn_projs = torch.randn( (bases.shape[0], num_rand_projs), 
                device=bases.device
            )
            randn_projs_norm = torch.norm(randn_projs, 2, dim=0, keepdim=True)
            randn_projs_scaled = randn_projs.div(randn_projs_norm)

            return torch.cat((bases, randn_projs_scaled), dim=-1)

        assert params_traj_samples
        subspace_dim = subspace_dim if subspace_dim is not None else len(params_traj_samples)
        params_mat = torch.moveaxis(
            torch.stack(params_traj_samples), (0, 1), (1, 0)
            ) # (num_params, subspace_dim)
        params_mat = params_mat if not use_cpu else params_mat.cpu()
        bases, singular_values, _  = tl.partial_svd(params_mat, n_eigenvecs=subspace_dim)
        
        if num_rand_projs is not None: 
            bases = _add_random_projs(
                bases=bases,
                num_rand_projs=num_rand_projs
            )
        
        """
        Returns
        -------
        bases : Tensor Size. (num_params, subspace_dim + num_rand_projs)
        """
        return bases.detach().to(device=device) if not return_singular_values else (
            bases.detach().to(device=device), singular_values.detach().to(device=device)
        )

    @classmethod
    def compute_traj_samples_mean(cls,
        params_traj_samples : List[Tensor],
        device = None, 
    ):
        params_mat = torch.moveaxis(
            torch.stack(params_traj_samples), (0, 1), (1, 0)
            )
        return params_mat.mean(dim=-1).to(device=device) # scalar

    def sample(self, 
        ray_trafo : BaseRayTrafo,
        dataset_kwargs : Dict, 
        optim_kwargs : Dict
        ):

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'SubspaceConstructor_Pretraining'
        logdir = os.path.join(
            optim_kwargs['log_path'],
            current_time + '_' + socket.gethostname() + '_' + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

        if optim_kwargs['torch_manual_seed']:
            torch.random.manual_seed(optim_kwargs['torch_manual_seed'])

        # create PyTorch datasets

        criterion = torch.nn.MSELoss()
        self.init_optimizer(optim_kwargs=optim_kwargs)

        dataset_train = get_ellipses_dataset(
            ray_trafo=ray_trafo, 
            fold='train', 
            im_size=dataset_kwargs['im_size'], 
            length=dataset_kwargs['length'], 
            white_noise_rel_stddev=dataset_kwargs['white_noise_rel_stddev'], 
            use_fixed_seeds_starting_from=dataset_kwargs['use_fixed_seeds_starting_from'], 
            device=self.device
        )

        dataset_validation = get_ellipses_dataset(
            ray_trafo=ray_trafo, 
            fold='validation', 
            im_size=dataset_kwargs['im_size'],
            length=dataset_kwargs['length']//10, 
            white_noise_rel_stddev=dataset_kwargs['white_noise_rel_stddev'], 
            use_fixed_seeds_starting_from=dataset_kwargs['use_fixed_seeds_starting_from'], 
            device=self.device
        )

        # create PyTorch dataloaders
        data_loaders = {
            'train': DataLoader(
                dataset_train, 
                batch_size=optim_kwargs['batch_size'],
                shuffle=True
            ),
            'validation': DataLoader(
                dataset_validation, 
                batch_size=optim_kwargs['batch_size'],
                shuffle=False
            )
        }

        dataset_sizes = {'train': len(dataset_train), 'validation': len(dataset_validation)}

        num_overall_updates = ceil(
            dataset_sizes['train'] / optim_kwargs['batch_size']
            ) * optim_kwargs['epochs']

        sample_idx_sequence = np.linspace(
            optim_kwargs['burn_in'], 
            num_overall_updates, 
            optim_kwargs['num_samples'] + 1, 
            dtype=int
            )

        self.init_scheduler(optim_kwargs=optim_kwargs)
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))
        
        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = -np.inf

        self.model.to(self.device)
        self.model.train()

        num_grad_updates = 0
        for epoch in range(optim_kwargs['epochs']):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1) ) as pbar:
                    for _, gt, fbp in pbar:

                        fbp = fbp.to(self.device)
                        gt = gt.to(self.device)

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(fbp)
                            loss = criterion(outputs, gt)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self._optimizer.step()

                                if num_grad_updates in sample_idx_sequence:
                                    self.add_params_traj_samples()        

                                if (self._scheduler is not None and
                                        schedule_every_batch):
                                    self._scheduler.step()

                        for i in range(outputs.shape[0]):
                            gt_ = gt[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, gt_, data_range=1)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})

                        if phase == 'train':
                            num_grad_updates += 1
                            self.writer.add_scalar('loss', running_loss/running_size, num_grad_updates)
                            self.writer.add_scalar('psnr', running_psnr/running_size, num_grad_updates)
                            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], num_grad_updates)

                    if phase == 'train':
                        if (self._scheduler is not None
                                and not schedule_every_batch):
                            self._scheduler.step()
                    
                    
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if phase == 'validation':
                        self.writer.add_scalar('val_loss', epoch_loss, num_grad_updates)
                        self.writer.add_scalar('val_psnr', epoch_psnr, num_grad_updates)
                        self.writer.add_image('reco', normalize(
                            outputs[0, ].detach().cpu().numpy()
                            ), 
                            num_grad_updates
                        )

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if optim_kwargs['save_best_learned_params_path'] is not None:
                            self.save_learned_params(
                                optim_kwargs['save_best_learned_params_path'])

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)
        
        self.writer.close()
        self.save_params_traj_samples()

    def init_optimizer(self, optim_kwargs: Dict):
        """
        Initialize the optimizer.
        """
        self._optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optim_kwargs['optimizer']['lr'],
                weight_decay=optim_kwargs['optimizer']['weight_decay'])

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def init_scheduler(self, optim_kwargs: Dict):
        if optim_kwargs['scheduler']['name'].lower() == 'cosine':
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=optim_kwargs['epochs'],
                eta_min=optim_kwargs['scheduler']['lr_min'])
        elif optim_kwargs['scheduler']['name'].lower() == 'onecyclelr':
            self._scheduler = OneCycleLR(
                self.optimizer,
                steps_per_epoch=ceil( optim_kwargs['scheduler']['train_len'] /  optim_kwargs['batch_size']),
                max_lr= optim_kwargs['scheduler']['max_lr'],
                epochs= optim_kwargs['epochs'])
        else:
            raise KeyError

    @property
    def scheduler(self):
        """
        torch learning rate scheduler :
        The scheduler, usually set by :meth:`init_scheduler`, which gets called
        in :meth:`train`.
        """
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value

    def save_learned_params(self, path):
        """
        Save learned parameters from file.
        """
        path = path if path.endswith('.pt') else path + 'nn_learned_params.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_learned_params(self, path):
        """
        Load learned parameters from file.
        """
        # TODO: not suitable for nn.DataParallel
        path = path if path.endswith('.pt') else path + '.pt'
        map_location = ('cuda:0' if self.use_cuda and torch.cuda.is_available()
                        else 'cpu')
        state_dict = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state_dict)
