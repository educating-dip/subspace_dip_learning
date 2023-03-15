"""
Code re-adapted from 
https://github.com/rb876/deep_image_prior_extension/blob/4e8e118718f51e4eeb0bb7be093959fecd80a561/src/dataset/pascal_voc.py
Provides the PascalVOCDataset.
"""
from typing import Optional
import numpy as np
import torch
from odl import uniform_discr

from ..simulation import SimulatedDataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Grayscale, RandomCrop, PILToTensor, Lambda, Compose

class PascalVOCDataset(torch.utils.data.IterableDataset):
    """
    Dataset with randomly cropped patches from Pascal VOC2012
    (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
    """
    def __init__(self,
            data_path: str, 
            year: str = '2012', 
            shuffle: bool = True,
            fold: = 'train', 
            im_size: int = 128,
            fixed_seeds: bool = True
        ):

        self.shape = (im_size, im_size)
        min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        max_pt = [self.shape[0]/2, self.shape[1]/2]
        self.space = uniform_discr(min_pt, max_pt, self.shape)

        self.transform = Compose(
                [Grayscale(),
                 RandomCrop(
                        size=im_size, padding=True, pad_if_needed=True, padding_mode='reflect'),
                 PILToTensor(),
                 Lambda(lambda x: ((x.to(torch.float32) + torch.rand(*x.shape)) / 256).numpy()),
                ])
    
        self.datasets = {
            'train': VOCSegmentation(
                root=data_path, year=year, image_set='train'),
            'validation': VOCSegmentation(
                root=data_path, year=year, image_set='val')
            }

        if isinstance(shuffle, bool):
            self.shuffle = {
                    'train': shuffle, 'validation': shuffle}
        else:
            self.shuffle = shuffle.copy()
        if isinstance(fixed_seeds, bool):
            if fixed_seeds:
                self.fixed_seeds = {'train': 1, 'validation': 2}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()
        
        self.rng = np.random.RandomState(
                self.fixed_seeds.get(fold, None)
            )

    def _generate_item(self, fold, idx):
        image = self.datasets[fold][idx][0]
        seed = self.rng.randint(np.iinfo(np.int64).max)
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            image = self.transform(image)[0, :, :]
        image -= image.min()
        image /= image.max()
        return image

    def generator(self, fold='train'):
        idx_list = self.rng.randint(len(self.datasets[fold]), size=self.get_len(fold))
        if self.shuffle[fold]:
            self.rng.shuffle(idx_list)
        for idx in idx_list:
            yield self._generate_item(fold, idx)

def get_pascal_voc_dataset(
        ray_trafo: BaseRayTrafo, 
        data_path : str, 
        im_size : int = 128,
        fold : str = 'train',
        white_noise_rel_stddev : float = .05, 
        use_fixed_seeds_starting_from : int = 1, 
        device : Optional[Any] = None) -> SimulatedDataset:

    image_dataset = PascalVOCDataset(
            data_path=data_path, 
            im_size = im_size,
            fold=fold
            )
    
    return SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=white_noise_rel_stddev,
            use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
            device=device
        )