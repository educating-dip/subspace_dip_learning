from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from subspace_dip.utils import PSNR, SSIM
from subspace_dip.dip import DeepImagePrior, SubspaceDeepImagePrior, SubspaceConstructor

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
    
    subspace_constructor = SubspaceConstructor(
        model=base_reconstructor.nn_model,
        device=device
    )

    optim_kwargs = {
        'log_path': './',
        'seed': cfg.seed, 
        'torch_manual_seed':cfg.dip.torch_manual_seed,
        'save_best_learned_params_path': './',
        'epochs': 15,
        'num_samples': 200,
        'burn_in': 100,
        'batch_size': 16,
        'weight_decay': 1e-3,
        'optimizer': {
            'lr': 1e-3,
            'weight_decay': 1e-8,
        },
        'scheduler': {
            'name': 'cosine',
            'train_len': 3200,
            'lr_min': 5e-5,
            'max_lr': 5e-3
        }
    }

    dataset_kwargs = {
        'im_size': cfg.dataset.im_size,
        'length': cfg.dataset.length,
        'white_noise_rel_stddev': cfg.dataset.noise_stddev,
        'use_fixed_seeds_starting_from': cfg.seed, 
    }

    subspace_constructor.sample(
        ray_trafo=ray_trafo,
        dataset_kwargs=dataset_kwargs, 
        optim_kwargs=optim_kwargs
    )

    weights_subspace = subspace_constructor.compute_bases_span_subspace(
        params_traj_samples=subspace_constructor.params_traj_samples,
        subspace_dim=200, 
        device=device
    )
    mean_weights = subspace_constructor.compute_traj_samples_mean(
        params_traj_samples=subspace_constructor.params_traj_samples,
        device=device
    )

    dataset = get_standard_dataset(
            cfg, 
            ray_trafo, 
            use_fixed_seeds_starting_from=cfg.seed,
            device=device, 
            use_adp_dataset=True
        )

    # within subspace optimization 
    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        reconstructor = SubspaceDeepImagePrior(
                ray_trafo=ray_trafo,
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
                'lr': 1e-2,
                'weight_decay': 1e-8, 
                'iterations': 10000,
                'loss_function': cfg.dip.optim.loss_function,
                'gamma': cfg.dip.optim.gamma}

        recon = reconstructor.reconstruct(
                subspace = weights_subspace, 
                mean = mean_weights, 
                noisy_observation = observation,
                filtbackproj=filtbackproj,
                ground_truth=ground_truth,
                recon_from_randn=cfg.dip.recon_from_randn,
                log_path=cfg.dip.log_path,
                optim_kwargs=optim_kwargs)

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
