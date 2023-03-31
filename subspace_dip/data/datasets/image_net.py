"""
Provides the ImageNetDataset.
"""
from typing import Optional, Any, Iterator, Union
import os
import numpy as np
import torch
from odl import uniform_discr
from torch import Tensor

from ..simulation import SimulatedDataset
from ..trafo import BaseRayTrafo
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomCrop, PILToTensor, Lambda, Compose

class ImageNetDataset(torch.utils.data.IterableDataset):
    """
    Dataset with randomly cropped patches from ImageNet
    (https://www.image-net.org/)
    """
    def __init__(self,
            data_path: str, 
            shuffle: str = 'fixed_random_subset',  # 'all', 'first'
            fold: str = 'train', 
            im_size: int = 256,
            num_images: int = -1,
        ):

        self.shape = (im_size, im_size)
        min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        max_pt = [self.shape[0]/2, self.shape[1]/2]
        self.space = uniform_discr(min_pt, max_pt, self.shape)

        self.transform = Compose(
                [RandomCrop(
                        size=im_size, pad_if_needed=True, padding_mode='reflect'),
                 PILToTensor(),
                 Lambda(lambda x: ((x.to(torch.float32) + torch.rand(*x.shape)) / 256).numpy()),
                ])
        partition = {
            'train' : 'train', 
            'validation': 'val'
        }
        self.dataset = ImageFolder(
            root=os.path.join(data_path, partition[fold])
            )
        
        self.max_length = len(self.dataset)
        self.length = self.max_length if num_images == -1 else num_images
        assert shuffle in ('fixed_random_subset', 'all', 'first')
        self.shuffle = shuffle
        self.fixed_seed = {'train': 1, 'validation': 2}[fold]
        self.rng = np.random.RandomState(
                self.fixed_seed)
        if self.shuffle == 'fixed_random_subset':
            self.fixed_random_subset = np.arange(self.max_length)
            self.rng.shuffle(self.fixed_random_subset)
            self.fixed_random_subset = self.fixed_random_subset[:self.length]
    
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
        if self.shuffle == 'all':
            idx_list = np.arange(self.max_length)
            self.rng.shuffle(idx_list)
            idx_list = idx_list[:self.length]
        else:  # 'first', 'fixed_random_subset'
            idx_list = np.arange(self.length)
            self.rng.shuffle(idx_list)
            if self.shuffle == 'fixed_random_subset':
                self.fixed_random_subset[idx_list]
        for idx in idx_list:
            yield self._generate_item(idx)
    
    def __getitem__(self, idx: int) -> Tensor:
        if self.shuffle == 'all':
            idx = self.rng.randint(self.max_length)
        else:  # 'first', 'fixed_random_subset'
            idx = self.rng.randint(self.length)
            if self.shuffle == 'fixed_random_subset':
                idx = self.fixed_random_subset[idx]
        return self._generate_item(idx)

def get_image_net_dataset(
        ray_trafo: BaseRayTrafo, 
        data_path : str, 
        im_size : int = 256,
        fold : str = 'train',
        white_noise_rel_stddev : float = .05, 
        use_multi_stddev_white_noise : bool = False, 
        use_fixed_seeds_starting_from : int = 1,
        num_images : int = -1,
        device : Optional[Any] = None
        ) -> SimulatedDataset:

    image_dataset = ImageNetDataset(
            data_path = data_path, 
            im_size = im_size,
            fold = fold, 
            num_images = num_images,
            )
    
    return SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=white_noise_rel_stddev,
            use_multi_stddev_white_noise=use_multi_stddev_white_noise,
            use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
            device=device
        )
