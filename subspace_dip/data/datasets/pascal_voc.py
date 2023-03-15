"""
Code re-adapted from 
https://github.com/rb876/deep_image_prior_extension/blob/4e8e118718f51e4eeb0bb7be093959fecd80a561/src/dataset/pascal_voc.py
Provides the PascalVOCDataset.
"""
from typing import Optional, Any, Iterator, Union
import numpy as np
import torch
from odl import uniform_discr
from torch import Tensor

from ..simulation import SimulatedDataset
from ..trafo import BaseRayTrafo
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import RandomCrop, PILToTensor, Lambda, Compose

class PascalVOCDataset(torch.utils.data.IterableDataset):
    """
    Dataset with randomly cropped patches from Pascal VOC2012
    (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
    """
    def __init__(self,
            data_path: str, 
            year: str = '2012', 
            shuffle: bool = True,
            fold: str = 'train', 
            im_size: int = 128,
            fixed_seeds: bool = True,
            num_images: int = -1,
        ):

        self.shape = (im_size, im_size)
        min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        max_pt = [self.shape[0]/2, self.shape[1]/2]
        self.space = uniform_discr(min_pt, max_pt, self.shape)

        self.transform = Compose(
                [RandomCrop(
                        size=im_size, padding=True, pad_if_needed=True, padding_mode='reflect'),
                 PILToTensor(),
                 Lambda(lambda x: ((x.to(torch.float32) + torch.rand(*x.shape)) / 256).numpy()),
                ])
        partition = {
            'train' : 'train', 
            'validation': 'val'
        }
        self.dataset = VOCSegmentation(
                root=data_path, 
                year=year, 
                image_set=partition[fold], 
                # download=True
            )
        
        self.length = len(self.dataset.images) if num_images == -1 else num_images
        self.shuffle = shuffle
        self.fixed_seed = {'train': 1, 'validation': 2}[fold]
        self.rng = np.random.RandomState(
                self.fixed_seed)
    
    def __len__(self, ) -> Union[int, float]: 
        return self.length

    def _generate_item(self, idx):
        image = self.dataset[idx][0]
        seed = self.rng.randint(np.iinfo(np.int64).max)
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            image = self.transform(image)
        image -= image.min()
        image /= image.max()
        return torch.from_numpy(image).float()

    def  __iter__(self) -> Iterator[Tensor]:
        idx_list = np.arange(self.length)
        if self.shuffle:
            self.rng.shuffle(idx_list)
        for idx in idx_list:
            yield self._generate_item(idx)
    
    def __getitem__(self, idx: int) -> Tensor:
        if self.shuffle: 
            idx = self.rng.randint(self.length)
        return self._generate_item(idx)

def get_pascal_voc_dataset(
        ray_trafo: BaseRayTrafo, 
        data_path : str, 
        im_size : int = 128,
        fold : str = 'train',
        white_noise_rel_stddev : float = .05, 
        use_fixed_seeds_starting_from : int = 1, 
        num_images : int = -1,
        device : Optional[Any] = None) -> SimulatedDataset:

    image_dataset = PascalVOCDataset(
            data_path=data_path, 
            im_size = im_size,
            fold=fold, 
            num_images=num_images,
            )
    
    return SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=white_noise_rel_stddev,
            use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
            device=device
        )
