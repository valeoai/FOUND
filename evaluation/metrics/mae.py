"""
Code borrowed from SelfMask: https://github.com/NoelShin/selfmask
"""

import torch

def compute_mae(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    :param pred_mask: (H x W) or (B x H x W) a normalized prediction mask with values in [0, 1]
    :param gt_mask: (H x W) or (B x H x W) a binary ground truth mask with values in {0, 1}
    """
    return torch.mean(
        torch.abs(pred_mask - gt_mask.to(torch.float32)), dim=(-1, -2)
    ).cpu()
