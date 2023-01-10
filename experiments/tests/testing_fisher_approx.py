from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_test_dataset
from subspace_dip.dip import DeepImagePrior, SubspaceDeepImagePrior, LinearSubspace, FisherInfo

def diff_norm(A, B): 
    return torch.linalg.matrix_norm(A-B)

def rel_diff(A, B): 
    norm_A = torch.linalg.matrix_norm(A)
    return torch.linalg.matrix_norm(A-B) / norm_A

def angle_between_matrices(A, B):
    norm_A = torch.linalg.matrix_norm(A)
    norm_B = torch.linalg.matrix_norm(B)
    return torch.arccos( torch.trace(B.T@A) / (norm_A*norm_B) )

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

    dataset = get_standard_test_dataset(
        ray_trafo,
        dataset_kwargs=OmegaConf.to_object(cfg.test_dataset),
        ray_trafo_kwargs=OmegaConf.to_object(cfg.trafo),
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
        reconstructor.net_input = filtbackproj.to(dtype=dtype, device=reconstructor.device)
        
        fisher_info = FisherInfo(
            subspace_dip=reconstructor,
            init_damping=cfg.subspace.fisher_info.init_damping,
        )

        use_forward_op = True
        subspace.init_parameters()

        fisher_info.update(curvature_ema=0., num_random_vecs=None, use_forward_op=use_forward_op, mode='full') # using jacrev
        deterministic_fisher_matrix = fisher_info.matrix
        # constructing approx. Fisher
        fisher_info.update(curvature_ema=0., use_forward_op=use_forward_op, num_random_vecs=1200, mode='vjp_rank_one')
        random_fisher_matrix_using_all_probes = fisher_info.matrix
        fisher_info.update(curvature_ema=0., use_forward_op=use_forward_op, num_random_vecs=10, mode='vjp_rank_one')
        random_fisher_matrix_using_10_probes = fisher_info.matrix
        fisher_info.update(curvature_ema=0., use_forward_op=use_forward_op, num_random_vecs=100, mode='vjp_rank_one')
        random_fisher_matrix_using_100_probes = fisher_info.matrix

        
        print(f'||exact_Fisher||: {torch.linalg.matrix_norm(deterministic_fisher_matrix)}')
        print(f'||approx_Fisher||:  {torch.linalg.matrix_norm(random_fisher_matrix_using_all_probes)}')
        print(f'||approx_Fisher|| using 100 probes:  {torch.linalg.matrix_norm(random_fisher_matrix_using_100_probes)}')
        print(f'||approx_Fisher|| using 10 probes:  {torch.linalg.matrix_norm(random_fisher_matrix_using_10_probes)}')

        print(f'||exact_Fisher||/||approx_Fisher||: {torch.linalg.matrix_norm(deterministic_fisher_matrix) / torch.linalg.matrix_norm(random_fisher_matrix_using_all_probes)}')
        print(f'||exact_Fisher||/||approx_Fisher|| using 100 probes: {torch.linalg.matrix_norm(deterministic_fisher_matrix) / torch.linalg.matrix_norm(random_fisher_matrix_using_100_probes)}')
        print(f'||exact_Fisher||/||approx_Fisher|| using 10 probes: {torch.linalg.matrix_norm(deterministic_fisher_matrix) / torch.linalg.matrix_norm(random_fisher_matrix_using_10_probes)}')

        print(f'||exact_Fisher - approx_Fisher||: {diff_norm(deterministic_fisher_matrix, random_fisher_matrix_using_all_probes)}')
        print(f'||exact_Fisher - approx_Fisher|| using 100 probes: {diff_norm(deterministic_fisher_matrix, random_fisher_matrix_using_100_probes)}')
        print(f'||exact_Fisher - approx_Fisher|| using 10 probes: {diff_norm(deterministic_fisher_matrix, random_fisher_matrix_using_10_probes)}')

        print(f'||exact_Fisher - approx_Fisher||/||exact Fisher|| : {rel_diff(deterministic_fisher_matrix, random_fisher_matrix_using_all_probes)}')
        print(f'||exact_Fisher - approx_Fisher||/||exact Fisher|| using 100 probes: {rel_diff(deterministic_fisher_matrix, random_fisher_matrix_using_100_probes)}')
        print(f'||exact_Fisher - approx_Fisher||/||exact Fisher|| using 10 probes: {rel_diff(deterministic_fisher_matrix, random_fisher_matrix_using_10_probes)}')

        print(f' ∠(exact_Fisher, approx_Fisher):  {angle_between_matrices(deterministic_fisher_matrix, random_fisher_matrix_using_all_probes)}')
        print(f' ∠(exact_Fisher, approx_Fisher) using 100 probes: {angle_between_matrices(deterministic_fisher_matrix, random_fisher_matrix_using_100_probes)}')
        print(f' ∠(exact_Fisher, approx_Fisher): using 10 probes: {angle_between_matrices(deterministic_fisher_matrix, random_fisher_matrix_using_10_probes)}')

        probe = torch.randn(subspace.num_subspace_params, device=reconstructor.device)
        vF_exact = fisher_info.exact_cvp(v=probe, use_forward_op=use_forward_op, use_square_root=False) 
        vF_emp = fisher_info.ema_cvp(v=probe, use_square_root=False) # 100 probes are used here
        vF_emp_inv = fisher_info.ema_cvp(v=probe, use_square_root=False, use_inverse=True) # 100 probes are used here
        print(f'||vF_exact - vF_emp||: {torch.linalg.norm(vF_exact - vF_emp)}')

        # exact inverse test
        fisher_info.update(curvature_ema=0., num_random_vecs=None, use_forward_op=use_forward_op, mode='full')
        vF_exac_inv = fisher_info.ema_cvp(v=probe, use_square_root=False, use_inverse=True) # using jacrev assembled fisher
        print(f'||vF_exac_inv - vF_emp_inv||: {torch.linalg.norm(vF_exac_inv - vF_emp_inv)}')
        
        # testing computation correctness 
        vF_exact_2 = fisher_info.ema_cvp(v=probe, use_square_root=False)
        print(f'||vF_exact - vF_exact_2||:  {torch.linalg.norm(vF_exact - vF_exact_2)}')
        
if __name__ == '__main__':
    coordinator()
