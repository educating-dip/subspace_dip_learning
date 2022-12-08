from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_test_dataset
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
    
    fisher_info = FisherInfo(
            subspace_dip=reconstructor,
            init_damping=cfg.subspace.fisher_info.init_damping,
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
        reconstructor.net_input = filtbackproj.to(reconstructor.device)

        subspace.init_parameters()

        fisher_info.reset_fisher_matrix()
        fisher_info.update(curvature_ema=0., num_random_vecs=None, use_forward_op=True, mode='full')
        deterministic_fisher_matrix = fisher_info.matrix

        fisher_info.reset_fisher_matrix()
        fisher_info.update(curvature_ema=0., use_forward_op=True, num_random_vecs=500, mode='vjp_rank_one')
        random_fisher_matrix = fisher_info.matrix
        print(f'L2 {torch.sum((deterministic_fisher_matrix - random_fisher_matrix).pow(2))}')
        
        probe = torch.randn(30, device=reconstructor.device)
        vF_exact = fisher_info.exact_cvp(v=probe, use_forward_op=True, use_square_root=False)
        vF_emp = fisher_info.ema_cvp(v=probe, use_square_root=False)
        print(f'L2 {torch.sum((vF_exact - vF_emp).pow(2))}')
        
        fisher_info.reset_fisher_matrix()
        fisher_info.update(curvature_ema=0., num_random_vecs=None, use_forward_op=True, mode='full')
        vF_exact_2 = fisher_info.ema_cvp(v=probe, use_square_root=False)
        print(f'L2 {torch.sum((vF_exact - vF_exact_2).pow(2))}')

        
        
if __name__ == '__main__':
    coordinator()
