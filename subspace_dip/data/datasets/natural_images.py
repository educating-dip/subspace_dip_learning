"""
Provides the CartoonSetDataset.
"""
from typing import Tuple, Union, Iterator
import os
import glob
import torch
from torch import Tensor
from itertools import repeat
from PIL import Image, ImageOps
from torchvision import transforms

def ground_truth_images_paths(data_path: str = './'):
    # paths = glob.glob(
    #     os.path.join(data_path, '*.png')
    #     )
    paths = ['Baboon512.png', 'F16_512.png',  'House.png', 'Lena512.png', 'Peppers512.png']
    paths = [os.path.join(data_path, s) for s in paths]
    return iter(paths), len(paths)

def RGB_natural_images_to_tensor(image_path: str = './', 
    shape: Tuple[int, int] = (128, 128),
    ):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(shape)]
        )
    return transform(
        Image.open(image_path).convert('RGB')
        ).numpy()

class NaturalImagesMiniDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, 
        data_path: str = './',
        shape: Tuple[int, int] = (128, 128),
        ):

        super().__init__()
        self.shape = shape
        self.cartoonset, self.length = ground_truth_images_paths(
                data_path=data_path
            )
        self.natural_images_data = []
        
    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float('inf')
        
    def _extend_natural_images_data(self, min_length: int) -> None:

        n_to_generate = max(min_length - len(self.natural_images_data), 0)
        for _ in range(n_to_generate):
            image_path = next(self.cartoonset)
            cartoon = RGB_natural_images_to_tensor(
                    image_path=image_path,
                    shape=self.shape
                )
            self.natural_images_data.append(cartoon)

    def _generate_item(self, idx: int):
        image = self.natural_images_data[idx]
        return torch.from_numpy(image).float()

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_natural_images_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_natural_images_data(idx + 1)
        return self._generate_item(idx)
