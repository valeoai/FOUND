"""
Code borrowed from SelfMask: https://github.com/NoelShin/selfmask
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Union
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale

from datasets.utils import GaussianBlur
from datasets.geometric_transforms import (
    random_scale,
    random_crop,
    random_hflip,
)

def geometric_augmentations(
    image: Image.Image,
    random_scale_range: Optional[Tuple[float, float]] = None,
    random_crop_size: Optional[int] = None,
    random_hflip_p: Optional[float] = None,
    mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Image.Image, torch.Tensor]:
    """Note. image and mask are assumed to be of base size, thus share a spatial shape."""
    if random_scale_range is not None:
        image, mask = random_scale(
            image=image, random_scale_range=random_scale_range, mask=mask
        )

    if random_crop_size is not None:
        crop_size = (random_crop_size, random_crop_size)
        fill = tuple(np.array(image).mean(axis=(0, 1)).astype(np.uint8).tolist())
        image, offset = random_crop(image=image, crop_size=crop_size, fill=fill)

        if mask is not None:
            assert ignore_index is not None
            mask = random_crop(
                image=mask, crop_size=crop_size, fill=ignore_index, offset=offset
            )[0]

    if random_hflip_p is not None:
        image, mask = random_hflip(image=image, p=random_hflip_p, mask=mask)
    return image, mask

def photometric_augmentations(
    image: Image.Image,
    random_color_jitter: bool,
    random_grayscale: bool,
    random_gaussian_blur: bool,
    proba_photometric_aug: float,
) -> torch.Tensor:
    if random_color_jitter:
        color_jitter = ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )
        image = RandomApply([color_jitter], p=proba_photometric_aug)(image)

    if random_grayscale:
        image = RandomGrayscale(proba_photometric_aug)(image)

    if random_gaussian_blur:
        w, h = image.size
        image = GaussianBlur(kernel_size=int((0.1 * min(w, h) // 2 * 2) + 1))(
            image, proba_photometric_aug
        )
    return image