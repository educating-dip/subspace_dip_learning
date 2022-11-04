"""
Provides synthetic image datasets and access to stored datasets.
"""

from .rectangles import RectanglesDataset
from .ellipses import EllipsesDataset, get_ellipses_dataset
from .walnut_patches import WalnutPatchesDataset
from .cartoonset import CartoonSetDataset