"""
Provides the EllipsesDataset.
"""
from typing import Union, Iterator, Tuple 
import numpy as np
import torch 
from torch import Tensor
from itertools import repeat
from odl import uniform_discr
from odl.phantom import ellipsoid_phantom
from ..simulation import SimulatedDataset
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo

class EllipsesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random ellipses.
    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, 
            shape : Tuple[int, int] = (128,128), 
            length : int = 3200, 
            fixed_seed : int = 1, 
            fold : str = 'train'
        ):

        self.shape = shape
        min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        max_pt = [self.shape[0]/2, self.shape[1]/2]
        self.space = uniform_discr(min_pt, max_pt, self.shape)
        self.length = length
        self.ellipses_data = []
        self.setup_fold(
            fixed_seed=fixed_seed,
            fold=fold
        )
        super().__init__()

    def setup_fold(self, 
        fixed_seed : int = 1, 
        fold : str = 'train'
        ):

        fixed_seed = None if fixed_seed in [False, None] else int(fixed_seed)
        if (fixed_seed is not None) and (fold == 'validation'): 
            fixed_seed = fixed_seed + 1 
        self.rng = np.random.RandomState(
            fixed_seed
        )
        
    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float('inf')

    def _extend_ellipses_data(self, min_length: int) -> None:

        max_n_ellipse = 70
        ellipsoids = np.empty((max_n_ellipse, 6))
        n_to_generate = max(min_length - len(self.ellipses_data), 0)
        for _ in range(n_to_generate):
            v = (self.rng.uniform(-0.4, 1.0, (max_n_ellipse,)))
            a1 = .2 * self.rng.exponential(1., (max_n_ellipse,))
            a2 = .2 * self.rng.exponential(1., (max_n_ellipse,))
            x = self.rng.uniform(-0.9, 0.9, (max_n_ellipse,))
            y = self.rng.uniform(-0.9, 0.9, (max_n_ellipse,))
            rot = self.rng.uniform(0., 2 * np.pi, (max_n_ellipse,))
            n_ellipse = min(self.rng.poisson(40), max_n_ellipse)
            v[n_ellipse:] = 0.
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            image = ellipsoid_phantom(self.space, ellipsoids)
            # normalize the foreground (all non-zero pixels) to [0., 1.]
            image[np.array(image) != 0.] -= np.min(image)
            image /= np.max(image)

            self.ellipses_data.append(image.asarray())

    def _generate_item(self, idx: int) -> Tensor:
        
        image = self.ellipses_data[idx]
        return torch.from_numpy(image[None]).float()  # add channel dim

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_ellipses_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_ellipses_data(idx + 1)
        return self._generate_item(idx)


def get_ellipses_dataset(
        ray_trafo: BaseRayTrafo, 
        fold : str = 'train', 
        im_size : int = 128, 
        length : int = 3200, 
        white_noise_rel_stddev : float = .05, 
        use_fixed_seeds_starting_from : int = 1, 
        device = None) -> SimulatedDataset:

    image_dataset = EllipsesDataset(
            (im_size, im_size), 
            length=length,
            fold=fold, 
            )
    
    return SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=white_noise_rel_stddev,
            use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
            device=device
        )
