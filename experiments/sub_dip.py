from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from subspace_dip.data import get_ellipses_dataset
from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_test_dataset
from subspace_dip.utils import PSNR, SSIM
from subspace_dip.dip import DeepImagePrior, SubspaceDeepImagePrior, LinearSubspace, FisherInfoMat

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

    fisher_info_matrix = None 
    if cfg.subspace.subspace_fine_tuning_kwargs.optim.optimizer == 'ngd':
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

        subspace.set_paramerters_on_valset(       
            subspace_dip=reconstructor,
            ray_trafo=ray_trafo,
            valset=valset,
            optim_kwargs={
                'epochs': cfg.subspace.set_subspace_parameters_kwargs.epochs, 
                'batch_size': cfg.subspace.set_subspace_parameters_kwargs.batch_size,
                'optim':{
                    'lr': cfg.subspace.set_subspace_parameters_kwargs.optim.lr,
                    'weight_decay': cfg.subspace.set_subspace_parameters_kwargs.optim.weight_decay
                    },
                'log_path': './',
                'torch_manual_seed': cfg.dip.torch_manual_seed
                }
        )

        fisher_info_matrix = FisherInfoMat(
            subspace_dip=reconstructor,
            valset=valset, 
            im_shape=(cfg.test_dataset.im_size, cfg.test_dataset.im_size), 
            batch_size=cfg.subspace.fisher_info.batch_size
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
            'iterations': cfg.subspace.subspace_fine_tuning_kwargs.iterations,
            'loss_function': cfg.subspace.subspace_fine_tuning_kwargs.loss_function,
            'optim':{
                'lr': cfg.subspace.subspace_fine_tuning_kwargs.optim.lr,
                'weight_decay': cfg.subspace.subspace_fine_tuning_kwargs.optim.weight_decay, 
                'optimizer': cfg.subspace.subspace_fine_tuning_kwargs.optim.optimizer,
                'gamma': cfg.subspace.subspace_fine_tuning_kwargs.optim.gamma,
                'mixing_factor': cfg.subspace.subspace_fine_tuning_kwargs.optim.mixing_factor,
            }
        }

        subspace.init_parameters()
        recon = reconstructor.reconstruct(
            noisy_observation=observation,
            filtbackproj=filtbackproj,
            ground_truth=ground_truth,
            fisher_info_matrix=fisher_info_matrix,
            recon_from_randn=cfg.dip.recon_from_randn,
            log_path=cfg.dip.log_path,
            optim_kwargs=optim_kwargs
        )

        print('Subspace DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
