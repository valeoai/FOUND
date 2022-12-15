"""
Code borrowed from SelfMask: https://github.com/NoelShin/selfmask
"""

import torch

class FMeasure:
    def __init__(
        self,
        default_thres: float = 0.5,
        beta_square: float = 0.3,
        n_bins: int = 255,
        eps: float = 1e-7,
    ):
        """
        :param default_thres: a hyperparameter for F-measure that is used to binarize a predicted mask. Default: 0.5
        :param beta_square: a hyperparameter for F-measure. Default: 0.3
        :param n_bins: the number of thresholds that will be tested for F-max. Default: 255
        :param eps: a small value for numerical stability
        """

        self.beta_square = beta_square
        self.default_thres = default_thres
        self.eps = eps
        self.n_bins = n_bins

    def _compute_precision_recall(
        self, binary_pred_mask: torch.Tensor, gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        :param binary_pred_mask: (B x H x W) or (H x W)
        :param gt_mask: (B x H x W) or (H x W), should be the same with binary_pred_mask
        """
        tp = torch.logical_and(binary_pred_mask, gt_mask).sum(dim=(-1, -2))
        tp_fp = binary_pred_mask.sum(dim=(-1, -2))
        tp_fn = gt_mask.sum(dim=(-1, -2))

        prec = tp / (tp_fp + self.eps)
        recall = tp / (tp_fn + self.eps)
        return prec, recall

    def _compute_f_measure(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        thresholds: torch.Tensor = None,
    ) -> torch.Tensor:
        if thresholds is None:
            binary_pred_mask = pred_mask > self.default_thres
        else:
            binary_pred_mask = pred_mask > thresholds

        prec, recall = self._compute_precision_recall(binary_pred_mask, gt_mask)
        f_measure = ((1 + (self.beta_square**2)) * prec * recall) / (
            (self.beta_square**2) * prec + recall + self.eps
        )
        return f_measure.cpu()

    def _compute_f_max(
        self, pred_mask: torch.Tensor, gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute self.n_bins + 1  F-measures, each of which has a different threshold, then return the maximum
        F-measure among them.

        :param pred_mask: (H x W)
        :param gt_mask: (H x W)
        """

        # pred_masks, gt_masks: H x W -> self.n_bins x H x W
        pred_masks = pred_mask.unsqueeze(dim=0).repeat(self.n_bins, 1, 1)
        gt_masks = gt_mask.unsqueeze(dim=0).repeat(self.n_bins, 1, 1)

        # thresholds: self.n_bins x 1 x 1
        thresholds = (
            torch.arange(0, 1, 1 / self.n_bins)
            .view(self.n_bins, 1, 1)
            .to(pred_masks.device)
        )

        # f_measures: self.n_bins
        f_measures = self._compute_f_measure(pred_masks, gt_masks, thresholds)
        return torch.max(f_measures).cpu(), f_measures

    def _compute_f_mean(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        adaptive_thres = 2 * pred_mask.mean(dim=(-1, -2), keepdim=True)
        binary_pred_mask = pred_mask > adaptive_thres

        prec, recall = self._compute_precision_recall(binary_pred_mask, gt_mask)
        f_mean = ((1 + (self.beta_square**2)) * prec * recall) / (
            (self.beta_square**2) * prec + recall + self.eps
        )
        return f_mean.cpu()

    def __call__(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> dict:
        """
        :param pred_mask: (H x W) a normalized prediction mask with values in [0, 1]
        :param gt_mask: (H x W) a binary ground truth mask with values in {0, 1}
        :return: a dictionary with keys being "f_measure" and "f_max" and values being the respective values.
        """
        outputs: dict = dict()
        for k in ("f_measure", "f_mean"):
            outputs.update({k: getattr(self, f"_compute_{k}")(pred_mask, gt_mask)})

        f_max_, all_f = self._compute_f_max(pred_mask, gt_mask)
        outputs["f_max"] = f_max_
        outputs["all_f"] = all_f  # List of all f values for all thresholds
        return outputs
