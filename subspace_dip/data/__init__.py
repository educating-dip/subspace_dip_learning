"""
Provides data generation, data access, and the ray transform.
"""

from .datasets import (
        RectanglesDataset, 
        EllipsesDataset, 
        DiskDistributedEllipsesDataset, 
        LoDoPaBTorchDataset, 
        WalnutPatchesDataset, 
        CartoonSetDataset, 
        MayoDataset,
        NaturalImagesMiniDataset, 
        ImageNetDataset,
        get_ellipses_dataset, 
        get_disk_dist_ellipses_dataset, 
        get_lodopab_dataset,
        get_walnut_2d_observation,
        get_walnut_2d_ground_truth, 
        get_pascal_voc_dataset,
        get_image_net_dataset
        )

from .trafo import (
        BaseRayTrafo,
        MatmulRayTrafo,
        FanBeam2DRayTrafo,
        ParallelBeam2DRayTrafo,
        MultiBlurringTrafoIter,
        BlurringTrafo,
        IdentityTrafo,
        get_parallel_beam_2d_matmul_ray_trafo,
        get_odl_ray_trafo_parallel_beam_2d,
        get_odl_ray_trafo_parallel_beam_2d_matrix,
        get_odl_ray_trafo_fan_beam_2d_matrix,
        get_fan_beam_2d_matmul_ray_trafo
        )

from .simulation import simulate, SimulatedDataset
from .utils import get_ray_trafo
