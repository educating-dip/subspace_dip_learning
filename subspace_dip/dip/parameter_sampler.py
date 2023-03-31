"""
Provides :class:`ParameterSampler`.
"""
from typing import Dict, Optional, Any, Union

import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
import torch.nn as nn

from torch import Tensor
from copy import deepcopy
from math import ceil
from tqdm import tqdm
from torch.utils.data import DataLoader

from .incremental_svd import IncremetalSVD
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from subspace_dip.utils import (
    PSNR, normalize, get_original_cwd, get_params_from_nn_module, get_standard_training_dataset)

class ParameterSampler:
    """
    Wrapper for constructing a low-dimensional subspace of the NN optimisation trajectory.
    """
    def __init__(self, 
        model: nn.Module,
        exclude_norm_layers: bool = False, 
        include_bias: bool = True, 
        device: Optional[Any] = None,
        ):
        
        self.model = model
        self.exclude_norm_layers = exclude_norm_layers
        self.include_bias = include_bias
        self.parameters_samples = []
        self.device = device or torch.device(
                ('cuda:0' if torch.cuda.is_available() else 'cpu')
            )

    def add_parameters_samples(self, use_cpu: bool = True) -> None:
        parameter_vec = get_params_from_nn_module(
            self.model,
            exclude_norm_layers=self.exclude_norm_layers,
            include_bias=self.include_bias
            )
        self.parameters_samples.append(
            parameter_vec if not use_cpu else parameter_vec.cpu()
            )

    def reset_parameters_samples(self, ) -> None:
        self.parameters_samples = []

    def get_parameters_matrix_from_list(self, 
            return_numpy: bool = True
                ) -> Union[Tensor, np.asarray]:
        parameter_matrix = torch.stack(self.parameters_samples, dim=1)
        return parameter_matrix if not return_numpy else parameter_matrix.cpu().numpy()
    
    def create_sampling_sequence(self,
        burn_in: int, 
        num_overall_updates : int, 
        num_samples : int, 
        sampling_strategy : str = 'linear'
        ):
        if isinstance(sampling_strategy, str) == False or sampling_strategy not in ['linear', ]:
            import warnings.warn as warn
            warn('sampling strategy not recognised. defaulting to linear')
            sampling_strategy = 'linear'

        if sampling_strategy == 'linear':
            self.sampling_sequence =  np.linspace(
                burn_in,
                num_overall_updates,
                min(num_samples, num_overall_updates - burn_in) + 1, 
                dtype=int
                )
        else: 
            raise NotImplementedError
  
    def sample(self, 
        ray_trafo : BaseRayTrafo,
        dataset_kwargs : Dict, 
        optim_kwargs : Dict,
        save_samples: bool = False, 
        use_incremental_sampling: bool = False,
        incremental_sampling_kwargs: Optional[Dict] = None
        ):

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'sample'
        logdir = os.path.join(
            optim_kwargs['log_path'],
            current_time + '_' + socket.gethostname() + '_' + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)
        if optim_kwargs['torch_manual_seed']:
            torch.random.manual_seed(optim_kwargs['torch_manual_seed'])
        # create PyTorch datasets
        criterion = torch.nn.MSELoss()
        self.init_optimizer(optim_kwargs=optim_kwargs)
        dataset_train, dataset_validation = get_standard_training_dataset(
            ray_trafo=ray_trafo, 
            dataset_kwargs=dataset_kwargs, 
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
        self.init_scheduler(optim_kwargs=optim_kwargs)
        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = -np.inf
        num_overall_updates = ceil(
                dataset_sizes['train'] / optim_kwargs['batch_size']
            ) * optim_kwargs['epochs']
        self.create_sampling_sequence(
                burn_in=optim_kwargs['burn_in'], 
                num_overall_updates=num_overall_updates, 
                num_samples=optim_kwargs['num_samples'] + 1, 
                sampling_strategy=optim_kwargs['sampling_strategy']
            )
        self.use_incremental_sampling = use_incremental_sampling
        if use_incremental_sampling:
            self.incremental_svd = IncremetalSVD(**incremental_sampling_kwargs)

        self.model.to(self.device)
        self.model.train()
        num_grad_updates = 0
        for epoch in range(optim_kwargs['epochs']):
            if (    self.use_incremental_sampling and self.incremental_svd.stop     ): 
                break 
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
                        if (self.use_incremental_sampling and self.incremental_svd.stop
                                        and phase == 'train'): 
                            break
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
                                if self.use_incremental_sampling:
                                    if num_grad_updates == self.sampling_sequence[0]:
                                        self.add_parameters_samples()
                                        self.incremental_svd.start_tracking(
                                            data=self.get_parameters_matrix_from_list(return_numpy=True))
                                        self.reset_parameters_samples()
                                    elif num_grad_updates in self.sampling_sequence:
                                        self.add_parameters_samples()
                                        if len(self.parameters_samples) == self.incremental_svd.batch_size:
                                            self.incremental_svd.update(
                                                C=self.get_parameters_matrix_from_list(return_numpy=True))
                                            self.reset_parameters_samples()
                                else:
                                    if num_grad_updates in self.sampling_sequence:
                                        self.add_parameters_samples()
                                           
                        for i in range(outputs.shape[0]):
                            gt_ = gt[i, :].detach().cpu().numpy()
                            outputs_ = outputs[i, :].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, gt_)
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
                        if self._scheduler is not None:
                            self._scheduler.step()
                        
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]
                    if (phase == 'train' and (optim_kwargs['save_best_learned_params_path'] is not None) 
                            and optim_kwargs['save_best_learned_params_per_epoch'] and (epoch%100==0) ):            
                        self.save_learned_params(
                                optim_kwargs['save_best_learned_params_path'], 
                                comment=f'epoch_{epoch}_'
                            )
                    if phase == 'validation':
                        self.writer.add_scalar('val_loss', epoch_loss, num_grad_updates)
                        self.writer.add_scalar('val_psnr', epoch_psnr, num_grad_updates)
                        self.writer.add_image('reco', normalize(
                            outputs[0, ].detach().cpu().numpy()
                            ),
                            num_grad_updates
                            )
                        self.writer.add_image('ground_truth', normalize(
                            gt[0, ].detach().cpu().numpy()
                            ),
                            num_grad_updates
                            )
                        self.writer.add_image('fbp', normalize(
                            fbp[0, ].detach().cpu().numpy()
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
        if ( save_samples and not self.use_incremental_sampling ): self.save_sampled_parameters()
        if self.use_incremental_sampling: self.incremental_svd.save_ortho_basis()

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

    def save_learned_params(self, path, comment=None):
        """
        Save learned parameters from file.
        """
        if comment is not None: 
            path += comment
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

    def save_sampled_parameters(self, 
        name: str = 'parameters_samples',
        path: str = './'
        ):

        path = path if path.endswith('.pt') else path + name + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.parameters_samples, path)

    def load_sampled_paramters(self, 
        path_to_parameters_samples: str, 
        device: Optional[Any] = None
        ):
        
        path = os.path.join(get_original_cwd(), 
            path_to_parameters_samples if path_to_parameters_samples.endswith('.pt') \
                else path_to_parameters_samples + '.pt')
        self.parameters_samples.extend(
            torch.load(path, map_location=device)
        )
