"""
Provides :class:`SubspaceDeepImagePrior`.
"""
from typing import Optional, Union
import os
import socket
import datetime
from warnings import warn
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import functorch as ftch
import tensorboardX
from torch import Tensor
from torch.nn import MSELoss
from tqdm import tqdm
from subspace_dip.utils import tv_loss, PSNR, normalize
from subspace_dip.data import BaseRayTrafo
from .base_dip_image_prior import BaseDeepImagePrior

class SubspaceDeepImagePrior(BaseDeepImagePrior):

    def __init__(self,
            ray_trafo: BaseRayTrafo,
            torch_manual_seed: Union[int, None] = 1,
            device=None,
            net_kwargs=None):

        super().__init__(
            ray_trafo=ray_trafo,
            torch_manual_seed=torch_manual_seed,
            device=device,
            net_kwargs=net_kwargs
        )

        self.func_model_with_input, _ = ftch.make_functional(self.nn_model)

    def _get_func_params(self, 
            linear_coeffs : Tensor,
            subspace : Tensor, 
            mean: Tensor
        ):

        """
        Parameters
        ----------

        linear_coeffs : Tensor
            Parameters vector (`requires_grad=True`). Size. (subspace_dim)
        subspace : Tensor
            Bases defying subspace. Size. (num_params, subspace_dim)
        mean : Tensor
            NN parameters mean. Scalar. 
        """
        assert linear_coeffs[None, :].shape[-1] == subspace.shape[-1]

        weights = mean + (linear_coeffs[None, :] * subspace).sum(dim=-1) # sum over subspace_dim
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

    def reconstruct(self,
            subspace: Tensor, 
            mean: Tensor, 
            noisy_observation: Tensor,
            filtbackproj: Optional[Tensor] = None,
            ground_truth: Optional[Tensor] = None,
            recon_from_randn: bool = False,
            use_tv_loss: bool = True,
            log_path: str = '.',
            show_pbar: bool = True,
            optim_kwargs=None) -> Tensor:
        """
        Reconstruct (by "training" the DIP network).

        Parameters
        ----------
        subspace : Tensor
            Bases defying subspace. Size. (num_params, subspace_dim)   
        mean : Tensor
            NN parameters mean. Scalar.
        noisy_observation : Tensor
            Noisy observation. Shape: ``(1, 1, *self.ray_trafo.obs_shape)``.
        filtbackproj : Tensor, optional
            Filtered back-projection. Used as the network input if `recon_from_randn` is not `True`.
            Shape: ``(1, 1, *self.ray_trafo.im_shape)``
        ground_truth : Tensor, optional
            Ground truth. Used to print and log PSNR values.
            Shape: ``(1, 1, *self.ray_trafo.im_shape)``
        recon_from_randn : bool, optional
            If `True`, normal distributed noise with std-dev 0.1 is used as the network input;
            if `False` (the default), `filtbackproj` is used as the network input.
        use_tv_loss : bool, optional
            Whether to include the TV loss term.
            The default is `True`.
        log_path : str, optional
            Path for saving tensorboard logs. Each call to reconstruct creates a sub-folder
            in `log_path`, starting with the time of the reconstruction call.
            The default is `'.'`.
        show_pbar : bool, optional
            Whether to show a progress bar.
            The default is `True`.
        optim_kwargs : dict, optional
            Keyword arguments for optimization.
            The following arguments are supported:

            * `gamma` (float)
                Weighting factor of the TV loss term, the default is ``1e-4``.
            * `lr` (float)
                Learning rate, the default is ``1e-4``.
            * `iterations` (int)
                Number of iterations, the default is ``10000``.
            * `loss_function` (str)
                Discrepancy loss function, the default is ``'mse'``.

        Returns
        -------
        best_output : Tensor
            Model output with the minimum loss achieved during the training.
            Shape: ``(1, 1, *self.ray_trafo.im_shape)``.
        """

        writer = tensorboardX.SummaryWriter(
                logdir=os.path.join(log_path, '_'.join((
                        datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        socket.gethostname(),
                        'Subspace_DIP' if not use_tv_loss else 'Subspace_DIP+TV'))))

        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('gamma', 1e-4)
        optim_kwargs.setdefault('lr', 1e-4)
        optim_kwargs.setdefault('iterations', 10000)
        optim_kwargs.setdefault('loss_function', 'mse')

        self.set_nn_model_require_grad(False)

        self.nn_model.train()

        self.net_input = (
            0.1 * torch.randn(1, 1, *self.ray_trafo.im_shape, device=self.device)
            if recon_from_randn else
            filtbackproj.to(self.device))

        coeffs = nn.Parameter(
            torch.zeros(
                subspace.shape[-1],
                requires_grad=True,
                device=self.device
                )
            )
        
        self.optimizer = torch.optim.Adam(
            [coeffs],
            lr=optim_kwargs['lr'],
            weight_decay=optim_kwargs['weight_decay']
        )

        noisy_observation = noisy_observation.to(self.device)
        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        min_loss_state = {
            'loss': np.inf,
            'output': self.nn_model(self.net_input).detach(),  # pylint: disable=not-callable
            'params_state_dict': deepcopy(coeffs),
        }
               
        writer.add_image('filtbackproj', normalize(
               filtbackproj[0, ...]).cpu().numpy(), 0)

        with tqdm(range(optim_kwargs['iterations']), desc='DIP', disable=not show_pbar,
                miniters=optim_kwargs['iterations']//100) as pbar:

            for i in pbar:
                self.optimizer.zero_grad()
                func_params = self._get_func_params(coeffs, subspace, mean)
                output = self.func_model_with_input(func_params, self.net_input)
                loss = criterion(self.ray_trafo(output), noisy_observation)
                if use_tv_loss:
                    loss = loss + optim_kwargs['gamma'] * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1)

                if loss.item() < min_loss_state['loss']:
                    min_loss_state['loss'] = loss.item()
                    min_loss_state['output'] = output.detach()
                    min_loss_state['params_state_dict'] = deepcopy(coeffs)

                self.optimizer.step()

                for p in self.nn_model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if ground_truth is not None:
                    min_loss_output_psnr = PSNR(
                            min_loss_state['output'].detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(
                            output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description(f'DIP output_psnr={output_psnr:.1f}', refresh=False)
                    writer.add_scalar('min_loss_output_psnr', min_loss_output_psnr, i)
                    writer.add_scalar('output_psnr', output_psnr, i)

                writer.add_scalar('loss', loss.item(),  i)
                if i % 10 == 0:
                    writer.add_image('reco', normalize(
                            min_loss_state['output'][0, ...]).cpu().numpy(), i)

        self.nn_model.load_state_dict(min_loss_state['params_state_dict'])
        writer.close()

        return min_loss_state['output']
