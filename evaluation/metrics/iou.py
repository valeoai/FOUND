"""
Code adapted from SelfMask: https://github.com/NoelShin/selfmask
"""

from typing import Optional, Union

import numpy as np
import torch


def compute_iou(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    threshold: Optional[float] = 0.5,
    eps: float = 1e-7,
) -> Union[np.ndarray, torch.Tensor]:
    """
    :param pred_mask: (B x H x W) or (H x W)
    :param gt_mask: (B x H x W) or (H x W), same shape with pred_mask
    :param threshold: a binarization threshold
    :param eps: a small value for computational stability
    :return: (B) or (1)
    """
    assert pred_mask.shape == gt_mask.shape, f"{pred_mask.shape} != {gt_mask.shape}"
    # assert 0. <= pred_mask.to(torch.float32).min() and pred_mask.max().to(torch.float32) <= 1., f"{pred_mask.min(), pred_mask.max()}"

    if threshold is not None:
        pred_mask = pred_mask > threshold
    if isinstance(pred_mask, np.ndarray):
        intersection = np.logical_and(pred_mask, gt_mask).sum(axis=(-1, -2))
        union = np.logical_or(pred_mask, gt_mask).sum(axis=(-1, -2))
        ious = intersection / (union + eps)
    else:
        intersection = torch.logical_and(pred_mask, gt_mask).sum(dim=(-1, -2))
        union = torch.logical_or(pred_mask, gt_mask).sum(dim=(-1, -2))
        ious = (intersection / (union + eps)).cpu()
    return ious
