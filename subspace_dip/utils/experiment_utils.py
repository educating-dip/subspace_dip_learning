from torch.utils.data import Dataset
from subspace_dip.data import get_ray_trafo, SimulatedDataset
from subspace_dip.data import (
        RectanglesDataset, EllipsesDataset, WalnutPatchesDataset
    )

def get_standard_ray_trafo(cfg):
    kwargs = {}
    kwargs['angular_sub_sampling'] = cfg.trafo.angular_sub_sampling
    if cfg.dataset.name in ('ellipses', 'rectangles', 'walnut_patches'):
        kwargs['im_shape'] = (cfg.dataset.im_size, cfg.dataset.im_size)
        kwargs['num_angles'] = cfg.trafo.num_angles
    else:
        raise ValueError
    return get_ray_trafo(cfg.dataset.name, kwargs=kwargs)

def get_standard_dataset(cfg, ray_trafo, use_fixed_seeds_starting_from=1, device=None) -> Dataset:
    """
    Returns a dataset of tuples ``noisy_observation, x, filtbackproj``, where
        * `noisy_observation` has shape ``(1,) + obs_shape``
        * `x` is the ground truth (label) and has shape ``(1,) + im_shape``
        * ``filtbackproj = FBP(noisy_observation)`` has shape ``(1,) + im_shape``

    Parameters
    ----------
    use_fixed_seeds_starting_from : int, optional
        Fixed seed for noise generation, only used in simulated datasets.
    device : str or torch.device, optional
        If specified, data will be moved to the device. `ray_trafo`
        (including `ray_trafo.fbp`) must support tensors on the device.
    """

    name = cfg.dataset.name 

    if name == 'ellipses':

        image_dataset = EllipsesDataset(
                (cfg.dataset.im_size, cfg.dataset.im_size), 
                length=cfg.dataset.length,
                )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif name == 'rectangles':

        image_dataset = RectanglesDataset(
                (cfg.dataset.im_size, cfg.dataset.im_size),
                num_rects=cfg.dataset.num_rects,
                num_angle_modes=cfg.dataset.num_angle_modes,
                angle_modes_sigma=cfg.dataset.angle_modes_sigma)
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif name == 'walnut_patches':

        image_dataset = WalnutPatchesDataset(
            data_path=cfg.dataset.data_path_test, shape=(
                cfg.dataset.im_size, cfg.dataset.im_size
                ),
            walnut_id=cfg.dataset.walnut_id, orbit_id=cfg.dataset.orbit_id, 
            slice_ind=cfg.dataset.slice_ind, 
            )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    else:
        raise ValueError

    return dataset
