"""
Code adapted from SelfMask: https://github.com/NoelShin/selfmask
"""

from random import randint, random, uniform
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode as IM


def random_crop(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    crop_size: Tuple[int, int],  # (h, w)
    fill: Union[int, Tuple[int, int, int]],  # an unsigned integer or RGB,
    offset: Optional[Tuple[int, int]] = None,  # (top, left) coordinate of a crop
):
    assert type(crop_size) in (tuple, list) and len(crop_size) == 2

    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
        h, w = image.shape[-2:]
    elif isinstance(image, Image.Image):
        w, h = image.size
    elif isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        raise TypeError(type(image))

    pad_h, pad_w = max(crop_size[0] - h, 0), max(crop_size[1] - w, 0)

    image = TF.pad(image, [0, 0, pad_w, pad_h], fill=fill, padding_mode="constant")

    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        h, w = image.shape[-2:]

    if offset is None:
        offset = (randint(0, h - crop_size[0]), randint(0, w - crop_size[1]))

    image = TF.crop(
        image, top=offset[0], left=offset[1], height=crop_size[0], width=crop_size[1]
    )
    return image, offset


def compute_size(
    input_size: Tuple[int, int], output_size: int, edge: str  # h, w
) -> Tuple[int, int]:
    assert edge in ["shorter", "longer"]
    h, w = input_size

    if edge == "longer":
        if w > h:
            h = int(float(h) / w * output_size)
            w = output_size
        else:
            w = int(float(w) / h * output_size)
            h = output_size
        assert w <= output_size and h <= output_size

    else:
        if w > h:
            w = int(float(w) / h * output_size)
            h = output_size
        else:
            h = int(float(h) / w * output_size)
            w = output_size
        assert w >= output_size and h >= output_size
    return h, w


def resize(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    size: Union[int, Tuple[int, int]],
    interpolation: str,
    edge: str = "both",
) -> Union[Image.Image, torch.Tensor]:
    """
    :param image: an image to be resized
    :param size: a resulting image size
    :param interpolation: sampling mode. ["nearest", "bilinear", "bicubic"]
    :param edge: Default: "both"
    No-op if a size is given as a tuple (h, w).
    If set to "both", resize both height and width to the specified size.
    If set to "shorter", resize the shorter edge to the specified size keeping the aspect ratio.
    If set to "longer", resize the longer edge to the specified size keeping the aspect ratio.
    :return: a resized image
    """
    assert interpolation in ["nearest", "bilinear", "bicubic"], ValueError(
        interpolation
    )
    assert edge in ["both", "shorter", "longer"], ValueError(edge)
    interpolation = {
        "nearest": IM.NEAREST,
        "bilinear": IM.BILINEAR,
        "bicubic": IM.BICUBIC,
    }[interpolation]

    if type(image) == torch.Tensor:
        image = image.clone().detach()
    elif type(image) == np.ndarray:
        image = torch.from_numpy(image)

    if type(size) is tuple:
        if type(image) == torch.Tensor and len(image.shape) == 2:
            image = TF.resize(
                image.unsqueeze(dim=0), size=size, interpolation=interpolation
            ).squeeze(dim=0)
        else:
            image = TF.resize(image, size=size, interpolation=interpolation)

    else:
        if edge == "both":
            image = TF.resize(image, size=[size, size], interpolation=interpolation)

        else:
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[-2:]
            rh, rw = compute_size(input_size=(h, w), output_size=size, edge=edge)
            image = TF.resize(image, size=[rh, rw], interpolation=interpolation)
    return image


def random_scale(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    random_scale_range: Tuple[float, float],
    mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
):
    scale = uniform(*random_scale_range)
    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        h, w = image.shape[-2:]
    w_rs, h_rs = int(w * scale), int(h * scale)
    image: Image.Image = resize(image, size=(h_rs, w_rs), interpolation="bilinear")
    if mask is not None:
        mask = resize(mask, size=(h_rs, w_rs), interpolation="nearest")
    return image, mask


def random_hflip(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    p: float,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
):
    assert 0.0 <= p <= 1.0, ValueError(random_hflip)

    # Return a random floating point number in the range [0.0, 1.0).
    if random() > p:
        image = TF.hflip(image)
        if mask is not None:
            mask = TF.hflip(mask)
    return image, mask
