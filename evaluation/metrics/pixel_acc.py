"""
Code borrowed from SelfMask: https://github.com/NoelShin/selfmask
"""

from typing import Optional

import torch


def compute_pixel_accuracy(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: Optional[float] = 0.5
) -> torch.Tensor:
    """
    :param pred_mask: (H x W) or (B x H x W) a normalized prediction mask with values in [0, 1]
    :param gt_mask: (H x W) or (B x H x W) a binary ground truth mask with values in {0, 1}
    """
    if threshold is not None:
        binary_pred_mask = pred_mask > threshold
    else:
        binary_pred_mask = pred_mask
    return (binary_pred_mask == gt_mask).to(torch.float32).mean(dim=(-1, -2)).cpu()
