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
    
    if cfg.load_dip_models_from_path is not None: 
        base_reconstructor.load_pretrain_model(
            learned_params_path=cfg.load_dip_models_from_path)
            
    subspace_constructor = SubspaceConstructor(
            model=base_reconstructor.nn_model,
            device=device
        )

    optim_kwargs = {
        'log_path': './',
        'seed': cfg.seed, 
        'torch_manual_seed': cfg.dip.torch_manual_seed,
        'save_best_learned_params_path': './',
        'save_best_learned_params_per_epoch': cfg.subspace.training.save_best_learned_params_per_epoch,
        'epochs': cfg.subspace.training.num_epochs,
        'num_samples': cfg.subspace.sampling.num_samples,
        'burn_in': cfg.subspace.sampling.burn_in,
        'batch_size': cfg.subspace.training.batch_size,
        'optimizer': {
            'lr': cfg.subspace.training.optim.lr,
            'weight_decay': cfg.subspace.training.optim.weight_decay,
        },
        'scheduler': {
            'name': 'cosine',
            'train_len': cfg.dataset.length,
            'lr_min': cfg.subspace.training.optim.lr_min,
            'max_lr': cfg.subspace.training.optim.max_lr
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

if __name__ == '__main__':
    coordinator()
