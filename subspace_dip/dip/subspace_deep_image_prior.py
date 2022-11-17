"""
Provides :class:`SubspaceDeepImagePrior`.
"""
from typing import Optional, Union, Tuple, Dict, Sequence
import os
import socket
import datetime
import torch
import numpy as np
import functorch as ftch
import torch.nn as nn
import tensorboardX

from warnings import warn
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from subspace_dip.utils import tv_loss, PSNR, SSIM, normalize
from subspace_dip.data import BaseRayTrafo
from .base_dip_image_prior import BaseDeepImagePrior
from .linear_subspace import LinearSubspace
from .fisher_info import FisherInfo
from .natural_gradient_optim import NGD

class SubspaceDeepImagePrior(BaseDeepImagePrior, nn.Module):

    def __init__(self,
        subspace: LinearSubspace,
        ray_trafo: BaseRayTrafo,
        state_dict: Optional[None] = None, 
        torch_manual_seed: Union[int, None] = 1,
        device=None,
        net_kwargs=None,
        ):

        nn.Module.__init__(self, )
        BaseDeepImagePrior.__init__(self, 
            ray_trafo=ray_trafo,
            torch_manual_seed=torch_manual_seed,
            device=device,
            net_kwargs=net_kwargs
        )
        
        self.subspace = subspace
        if state_dict is not None: 
            self.nn_model.load_state_dict(
                state_dict=state_dict
            )
        self.func_model_with_input, _ = ftch.make_functional(self.nn_model)
        self.pretrained_weights = torch.cat(
            [param.flatten().detach() for param in self.nn_model.parameters()]
        )

    def get_func_params(self, 
            parameters_vec: Optional[Tensor] = None,
            slicing_sequence: Optional[Sequence] = None, 
        ) -> Tuple[Tensor]:

        weights = self.pretrained_weights
        if slicing_sequence is None:
            weights = weights + torch.inner(
                    self.subspace.parameters_vec if parameters_vec is None \
                        else parameters_vec, self.subspace.ortho_basis
                    )
        else:
            weights = weights + torch.inner(
                    self.subspace.parameters_vec[slicing_sequence] if parameters_vec is None \
                        else parameters_vec[slicing_sequence], self.subspace.ortho_basis[:, slicing_sequence]
                    )
        cnt = 0
        func_weights = []
        for params in self.nn_model.parameters():
            func_weights.append(
                weights[cnt:cnt+params.numel()].view(params.shape)
            )
            cnt += params.numel()
        return tuple(func_weights)

    def set_nn_model_require_grad(self, set_require_grad: bool):
        for params in self.nn_model.parameters():
            params.requires_grad_(set_require_grad)

    def forward(self, 
            parameters_vec: Optional[Tensor] = None, 
            input: Optional[Tensor] = None, 
            slicing_sequence: Optional[Sequence] = None, 
            use_forward_op: bool = True
        ) -> Tensor:

        out = self.func_model_with_input(
                self.get_func_params(
                    parameters_vec=self.subspace.parameters_vec if parameters_vec is None else parameters_vec,
                    slicing_sequence=slicing_sequence
                        ), 
                self.net_input if input is None else input
            )
        return out if not use_forward_op else self.ray_trafo(out)

    def objective(self,
        criterion,
        noisy_observation: Tensor,
        use_tv_loss: Optional[bool] = None,
        slicing_sequence: Optional[Sequence] = None,
        use_forward_op: bool = False,
        gamma: Optional[float] = None, 
        return_output: Optional[bool] = True
        ):

        output = self.forward(
                slicing_sequence=slicing_sequence,
                use_forward_op=use_forward_op
            )

        loss = criterion(self.ray_trafo(output), noisy_observation)
        if use_tv_loss:
            loss = loss + gamma*tv_loss(output)
        return loss if not return_output else (loss, output)

    def reconstruct(self,
        noisy_observation: Tensor,
        filtbackproj: Optional[Tensor] = None,
        ground_truth: Optional[Tensor] = None,
        recon_from_randn: bool = False,
        use_tv_loss: bool = True,
        fisher_info: Optional[FisherInfo] = None,
        log_path: str = '.',
        show_pbar: bool = True,
        optim_kwargs: Dict = None
        ) -> Tensor:

        writer = tensorboardX.SummaryWriter(
                logdir=os.path.join(log_path, '_'.join((
                        datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        socket.gethostname(),
                        '_Subspace_DIP' if not use_tv_loss else '_Subspace_DIP+TV'))))

        self.set_nn_model_require_grad(False)
        self.nn_model.train()

        self.net_input = (
            0.1 * torch.randn(1, 1, *self.ray_trafo.im_shape, device=self.device)
            if recon_from_randn else
            filtbackproj.to(self.device)
        )

        if optim_kwargs['optim']['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                [self.subspace.parameters_vec],
                lr=optim_kwargs['optim']['lr'],
                weight_decay=optim_kwargs['optim']['weight_decay']
                )
        elif optim_kwargs['optim']['optimizer'] == 'ngd':
            self.optimizer = NGD(
                [self.subspace.parameters_vec],
                lr=optim_kwargs['optim']['lr'],
                weight_decay=optim_kwargs['optim']['weight_decay']
                )
        elif optim_kwargs['optim']['optimizer'] == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(
                [self.subspace.parameters_vec], lr=optim_kwargs['optim']['lr'],
            )
        else: 
            raise NotImplementedError

        noisy_observation = noisy_observation.to(self.device)
        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        min_loss_state = {
            'loss': np.inf,
            'output': self.nn_model(self.net_input).detach(),  # pylint: disable=not-callable
            'params_state_dict': self.subspace.state_dict(),
        }

        if ground_truth is not None:
            writer.add_image('ground_truth', normalize(
               ground_truth[0, ...]).cpu().numpy(), 0)

        if filtbackproj is not None: 
            writer.add_image('filtbackproj', normalize(
                filtbackproj[0, ...]).cpu().numpy(), 0)

        if optim_kwargs['optim']['use_subsampling_orthospace']:
            self.singular_probabilities = self.subspace.singular_values / torch.sum(self.subspace.singular_values)

        writer.add_image('base_recon', normalize(
               self.nn_model(self.net_input)[0, ...].detach().cpu().numpy()), 0)
        
        print('Pre-trained UNET reconstruction of sample')
        print('PSNR:', PSNR(self.nn_model(self.net_input)[0, 0].detach().cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(self.nn_model(self.net_input)[0, 0].detach().cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        with tqdm(range(
                optim_kwargs['iterations']), desc='DIP', disable=not show_pbar
            ) as pbar:

            for i in pbar:
                slicing_sequence = None
                if optim_kwargs['optim']['use_subsampling_orthospace']:
                    slicing_sequence = np.random.choice(range(
                        self.subspace.num_subspace_params), 
                        size=optim_kwargs['optim']['subsampling_orthospace_dim'],
                        replace=False, 
                        p=self.singular_probabilities.cpu().numpy()
                    )
                self.optimizer.zero_grad()
                loss, output = self.objective(
                    criterion=criterion,
                    noisy_observation=noisy_observation,
                    use_tv_loss=use_tv_loss,
                    slicing_sequence=slicing_sequence,
                    gamma=optim_kwargs['optim']['gamma']
                    )
                loss.backward()

                if loss.item() < min_loss_state['loss']:
                    min_loss_state['loss'] = loss.item()
                    min_loss_state['output'] = output.detach()
                    min_loss_state['params_state_dict'] = self.subspace.state_dict()

                if optim_kwargs['optim']['optimizer'] == 'adam':
                    self.optimizer.step()
                elif optim_kwargs['optim']['optimizer'] == 'ngd':
                    fisher_info.update(tuneset=DataLoader(
                            torch.utils.data.TensorDataset(
                                noisy_observation,
                                filtbackproj,
                                ground_truth
                                )
                            ),
                        mixing_factor=optim_kwargs['optim']['mixing_factor'], 
                        damping_factor=optim_kwargs['optim']['damping_factor'],
                        use_forward_op=True,
                        mode=optim_kwargs['optim']['mode']
                    )
                    self.optimizer.step(
                            fisher_info=fisher_info, 
                        )
                elif optim_kwargs['optim']['optimizer'] == 'lbfgs':
                    self.optimizer.step(
                        lambda: self.objective(
                            criterion=criterion,
                            noisy_observation=noisy_observation,
                            use_tv_loss=use_tv_loss,
                            gamma=optim_kwargs['optim']['gamma'],
                            return_output=False
                        )
                    )
                else: 
                    raise NotImplementedError

                for p in self.nn_model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if ground_truth is not None:
                    min_loss_output_psnr = PSNR(
                            min_loss_state['output'].detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(
                            output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description(f'DIP output_psnr={output_psnr:.3f}', refresh=False)
                    writer.add_scalar('min_loss_output_psnr', min_loss_output_psnr, i)
                    writer.add_scalar('output_psnr', output_psnr, i)

                writer.add_scalar('loss', loss.item(),  i)
                if i % 10 == 0:
                    writer.add_image('reco', normalize(
                            min_loss_state['output'][0, ...]).cpu().numpy(), i)

        self.subspace.load_state_dict(
            min_loss_state['params_state_dict']
            )
        writer.close()

        return min_loss_state['output']
