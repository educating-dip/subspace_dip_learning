from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from subspace_dip.utils import PSNR, SSIM
from subspace_dip.dip import DeepImagePrior, SubspaceDeepImagePrior, LinearSubspace

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ray_trafo = get_standard_ray_trafo(cfg)
    ray_trafo.to(dtype=dtype, device=device)

    net_kwargs = {
            'scales': cfg.dip.net.scales,
            'channels': cfg.dip.net.channels,
            'skip_channels': cfg.dip.net.skip_channels,
            'use_norm': cfg.dip.net.use_norm,
            'use_sigmoid': cfg.dip.net.use_sigmoid,
            'sigmoid_saturation_thresh': cfg.dip.net.sigmoid_saturation_thresh
        }

    base_reconstructor = DeepImagePrior(
                ray_trafo, 
                torch_manual_seed=cfg.dip.torch_manual_seed,
                device=device, 
                net_kwargs=net_kwargs
            )
    
    base_reconstructor.load_pretrain_model(
        learned_params_path=cfg.load_dip_models_from_path)

    subspace = LinearSubspace(
        subspace_dim=cfg.subspace.subspace_dim,
        use_random_init=cfg.subspace.use_random_init,
        num_random_projs=cfg.subspace.num_random_projs,
        load_ortho_basis_path=cfg.subspace.ortho_basis_path,
        device=device
    )

    dataset = get_standard_dataset(
            cfg, 
            ray_trafo, 
            use_fixed_seeds_starting_from=cfg.seed,
            device=device, 
        )

    # within subspace optimization 
    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        reconstructor = SubspaceDeepImagePrior(
            ray_trafo=ray_trafo,
            subspace=subspace,
            state_dict=base_reconstructor.nn_model.state_dict(),
            torch_manual_seed=cfg.dip.torch_manual_seed,
            device=device, 
            net_kwargs=net_kwargs
            )
    
        observation, ground_truth, filtbackproj = data_sample

        observation = observation.to(dtype=dtype, device=device)
        filtbackproj = filtbackproj.to(dtype=dtype, device=device)
        ground_truth = ground_truth.to(dtype=dtype, device=device)

        optim_kwargs = {
            'lr': cfg.subspace.optim.lr,
            'weight_decay': cfg.subspace.optim.weight_decay, 
            'iterations': cfg.subspace.optim.iterations,
            'loss_function': cfg.subspace.optim.loss_function,
            'optimizer': cfg.subspace.optim.optimizer,
            'gamma': cfg.subspace.optim.gamma
            }

        recon = reconstructor.reconstruct(
                noisy_observation=observation,
                filtbackproj=filtbackproj,
                ground_truth=ground_truth,
                recon_from_randn=cfg.dip.recon_from_randn,
                log_path=cfg.dip.log_path,
                optim_kwargs=optim_kwargs
            )

        print('Subspace DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
