from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from subspace_dip.data import get_ellipses_dataset
from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_test_dataset
from subspace_dip.utils import PSNR, SSIM
from subspace_dip.dip import DeepImagePrior, SubspaceDeepImagePrior, LinearSubspace, FisherInfo

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    assert cfg.test_dataset.im_size == cfg.source_dataset.im_size

    ray_trafo = get_standard_ray_trafo(
        ray_trafo_kwargs=OmegaConf.to_object(cfg.trafo), 
        dataset_kwargs={
            'name': cfg.test_dataset.name,
            'im_size': cfg.test_dataset.im_size 
        }
    )
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

    reconstructor = SubspaceDeepImagePrior(
        ray_trafo=ray_trafo,
        subspace=subspace,
        state_dict=base_reconstructor.nn_model.state_dict(),
        torch_manual_seed=cfg.dip.torch_manual_seed,
        device=device, 
        net_kwargs=net_kwargs
    )

    fisher_info = None 
    if cfg.subspace.fine_tuning.optim.optimizer == 'ngd':

        fisher_info = FisherInfo(
            subspace_dip=reconstructor,
            init_damping=cfg.subspace.fisher_info.init_damping,
            curvature_ema=cfg.subspace.fisher_info.curvature_ema,
            sampling_probes_mode=cfg.subspace.fisher_info.sampling_probes_mode
        )
        if cfg.subspace.fisher_info.use_init_fisher_info_matrix:

            valset = DataLoader(
                get_ellipses_dataset(
                    ray_trafo=ray_trafo, 
                    fold='test', 
                    im_size=cfg.source_dataset.im_size,
                    length=cfg.source_dataset.length.test, 
                    white_noise_rel_stddev=cfg.source_dataset.noise_stddev, 
                    use_fixed_seeds_starting_from=cfg.seed, 
                    device=device
                ),
                batch_size=cfg.subspace.fisher_info.batch_size,
                shuffle=True
            )

            subspace.set_parameters_on_valset(       
                subspace_dip=reconstructor,
                ray_trafo=ray_trafo,
                valset=valset,
                optim_kwargs={
                    'epochs': cfg.subspace.fisher_info.init_fisher_info_matrix.epochs, 
                    'batch_size': cfg.subspace.fisher_info.init_fisher_info_matrix.batch_size,
                    'optim':{
                        'lr': cfg.subspace.fisher_info.init_fisher_info_matrix.optim.lr,
                        'weight_decay': cfg.subspace.fisher_info.init_fisher_info_matrix.optim.weight_decay, 
                        'gamma': cfg.subspace.fisher_info.init_fisher_info_matrix.optim.gamma
                        },
                    'log_path': './',
                    'torch_manual_seed': cfg.dip.torch_manual_seed
                    }
                )
            
            fisher_info.initialise_fisher_info(
                dataset=valset, 
                num_random_vecs=cfg.subspace.fisher_info.init_fisher_info_matrix.num_random_vecs,
                mode=cfg.subspace.fisher_info.mode,
            )

    dataset = get_standard_test_dataset(
        ray_trafo,
        dataset_kwargs=OmegaConf.to_object(cfg.test_dataset),
        use_fixed_seeds_starting_from=cfg.seed,
        device=device,
    )

    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        observation, ground_truth, filtbackproj = data_sample

        observation = observation.to(dtype=dtype, device=device)
        filtbackproj = filtbackproj.to(dtype=dtype, device=device)
        ground_truth = ground_truth.to(dtype=dtype, device=device)

        optim_kwargs = {
            'iterations': cfg.subspace.fine_tuning.iterations,
            'loss_function': cfg.subspace.fine_tuning.loss_function,
            'optim':{
                    'lr': cfg.subspace.fine_tuning.optim.lr,
                    'optimizer': cfg.subspace.fine_tuning.optim.optimizer,
                    'gamma': cfg.subspace.fine_tuning.optim.gamma,
                    'weight_decay': cfg.subspace.fine_tuning.optim.weight_decay,
                    'momentum': cfg.subspace.fine_tuning.optim.momentum,
                    'use_subsampling_orthospace': cfg.subspace.use_subsampling_orthospace,
                    'subsampling_orthospace_dim': cfg.subspace.subsampling_orthospace.subsampling_orthospace_dim
                }
        }

        if cfg.subspace.fine_tuning.optim.optimizer == 'ngd': 
            optim_kwargs['optim'].update(
            {
                'num_random_vecs': cfg.subspace.fisher_info.num_random_vecs,
                'use_adaptive_damping': cfg.subspace.fine_tuning.optim.use_adaptive_damping,
                'use_adaptive_learning_rate': cfg.subspace.fine_tuning.optim.use_adaptive_learning_rate,
                'use_adaptive_momentum': cfg.subspace.fine_tuning.optim.use_adaptive_momentum,
                'stats_interval': cfg.subspace.fine_tuning.optim.stats_interval,
                'scale_curvature': cfg.subspace.fine_tuning.optim.scale_curvature,
                'use_approximate_quad_model': cfg.subspace.fine_tuning.optim.use_approximate_quad_model,
                'mode': cfg.subspace.fisher_info.mode,
                'update_curvature_ema': cfg.subspace.fisher_info.update_curvature_ema,
                'curvature_ema_kwargs': OmegaConf.to_object(cfg.subspace.fisher_info.curvature_ema_kwargs), 
                'return_stats': cfg.subspace.fisher_info.return_stats
            })
            fisher_info.reset_fisher_matrix()

        subspace.init_parameters()
        recon = reconstructor.reconstruct(
            noisy_observation=observation,
            filtbackproj=filtbackproj,
            ground_truth=ground_truth,
            fisher_info=fisher_info,
            recon_from_randn=cfg.dip.recon_from_randn,
            log_path=cfg.dip.log_path,
            optim_kwargs=optim_kwargs
        )

        print('Subspace DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
