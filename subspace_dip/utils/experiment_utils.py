from typing import Dict
from torch.utils.data import Dataset
from subspace_dip.data import get_ray_trafo, SimulatedDataset
from subspace_dip.data import (
        RectanglesDataset, EllipsesDataset, WalnutPatchesDataset, 
        CartoonSetDataset
    )

def get_standard_ray_trafo(ray_trafo_kwargs: dict, dataset_kwargs: Dict):
    kwargs = {}
    kwargs['angular_sub_sampling'] = ray_trafo_kwargs['angular_sub_sampling']
    if dataset_kwargs['name'] in ('ellipses', 'rectangles', 'walnut_patches', 'cartoonset'):
        kwargs['im_shape'] = (dataset_kwargs['im_size'], dataset_kwargs['im_size'])
        kwargs['num_angles'] = ray_trafo_kwargs['num_angles']
    else:
        raise ValueError
    return get_ray_trafo(dataset_kwargs['name'], kwargs=kwargs)

def get_standard_test_dataset(
        ray_trafo,
        dataset_kwargs,
        use_fixed_seeds_starting_from=1, 
        device=None
    ) -> Dataset:

    if dataset_kwargs['name'] == 'ellipses':

        image_dataset = EllipsesDataset(
                (dataset_kwargs['im_size'], dataset_kwargs['im_size']), 
                length=dataset_kwargs['length']['test'],
                )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif dataset_kwargs['name'] == 'rectangles':

        image_dataset = RectanglesDataset(
                (dataset_kwargs['im_size'], dataset_kwargs['im_size']),
                num_rects=dataset_kwargs['num_rects'],
                num_angle_modes=dataset_kwargs['num_angle_modes'],
                angle_modes_sigma=dataset_kwargs['angle_modes_sigma'])
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif dataset_kwargs['name'] == 'walnut_patches':

        image_dataset = WalnutPatchesDataset(
            data_path=dataset_kwargs['data_path_test'], shape=(
                dataset_kwargs['im_size'], dataset_kwargs['im_size']
                ),
            walnut_id=dataset_kwargs['walnut_id'], orbit_id=dataset_kwargs['orbit_id'], 
            slice_ind=dataset_kwargs['slice_ind'], 
            )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif dataset_kwargs['name'] == 'cartoonset':

        image_dataset = CartoonSetDataset(
            data_path=dataset_kwargs['data_path_test'], shape=(
                dataset_kwargs['im_size'], dataset_kwargs['im_size']
                )
            )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    else:
        raise ValueError

    return dataset
