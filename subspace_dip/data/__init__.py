"""
Provides data generation, data access, and the ray transform.
"""

from .datasets import (
        RectanglesDataset, EllipsesDataset, DiskDistributedEllipsesDataset, WalnutPatchesDataset, CartoonSetDataset, 
        get_ellipses_dataset, get_disk_dist_ellipses_dataset, get_walnut_2d_observation, get_walnut_2d_ground_truth)
from .trafo import (
        BaseRayTrafo, MatmulRayTrafo,
        get_odl_ray_trafo_parallel_beam_2d, ParallelBeam2DRayTrafo,
        get_odl_ray_trafo_parallel_beam_2d_matrix,
        get_parallel_beam_2d_matmul_ray_trafo)
from .simulation import simulate, SimulatedDataset
from .utils import get_ray_trafo
