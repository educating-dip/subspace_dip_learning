"""
Provides :class:`SubspaceConstructor`.
"""
from typing import Dict, Optional
import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
import torch.nn as nn 
from math import ceil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from subspace_dip.utils import PSNR, normalize
from subspace_dip.data import SimulatedDataset
from subspace_dip.utils import get_params_from_nn_module

class SubspaceConstructor:

    """
    Wrapper for constructing a low-dimensional subspace of the NN optimisation trajectory.
    """

    def __init__(self, 
            model : nn.Module,
            subspace_dim : Optional['int'] = None,  
            device = None, 
            ):
        
        self.model = model
        self.device = device or torch.device(
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.subspace_dim = subspace_dim 
        self._params_traj_samples = []
    
    def _collect_params_traj_sample(self, ):
        
        self._params_traj_samples.append(
            get_params_from_nn_module(
                self.model,
                exclude_norm_layers=False,
                include_bias=True
            )
        )

    @property
    def subspace(self,
            return_singular_values : bool = False, 
            use_cpu : bool = True
        ):

        subspace_dim = self.subspace_dim if self.subspace_dim is not None else len(self._params_traj_samples)
        params_mat = torch.moveaxis(
            torch.stack(self._params_traj_samples), (0, 1), (1, 0)
            ) # (self.subspace_dim, num_params)
        params_mat = params_mat if not use_cpu else params_mat.cpu()
        bases, singular_values, _ = torch.svd_lowrank(params_mat, q=subspace_dim)
        
        """
        Returns
        -------
        bases : Tensor Size. (num_params, subspace_dim)
        """
        return bases.to(device=self.device) if not return_singular_values else (
            bases.to(device=self.device), singular_values.to(device=self.device)
        )

    @property
    def mean(self, ):
        params_mat = torch.moveaxis(
            torch.stack(self._params_traj_samples), (0, 1), (1, 0)
            )
        return params_mat.mean(dim=-1) # scalar

    def sample(self, 
        dataset : SimulatedDataset, 
        optim_kwargs : Dict
        ):

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'SubspaceConstructor_Pretraining'
        logdir = os.path.join(
            optim_kwargs['log_path'],
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

        if optim_kwargs['torch_manual_seed']:
            torch.random.manual_seed(optim_kwargs['torch_manual_seed'])

        # create PyTorch datasets

        criterion = torch.nn.MSELoss()
        self.init_optimizer(optim_kwargs=optim_kwargs)

        # create PyTorch dataloaders
        data_loaders = {'train': DataLoader(dataset, batch_size=optim_kwargs['batch_size'],
            shuffle=True),
            }

        dataset_sizes = {'train': len(dataset)}

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

        self.model.to(self.device)
        self.model.train()

        num_grad_updates = 0
        for epoch in range(optim_kwargs['epochs']):
            # Each epoch has a training and validation phase
            for phase in ['train']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode

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
                                    self._collect_params_traj_sample()        

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
                            self.writer.add_image('reco', normalize(outputs[0,].detach().cpu().numpy()), num_grad_updates)

                    if phase == 'train':
                        if (self._scheduler is not None
                                and not schedule_every_batch):
                            self._scheduler.step()
        
        self.writer.close()

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
        path = path if path.endswith('.pt') else path + '.pt'
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
