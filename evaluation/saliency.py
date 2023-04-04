# Copyright 2022 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy import ndimage

from evaluation.metrics.average_meter import AverageMeter
from evaluation.metrics.f_measure import FMeasure
from evaluation.metrics.iou import compute_iou
from evaluation.metrics.mae import compute_mae
from evaluation.metrics.pixel_acc import compute_pixel_accuracy
from evaluation.metrics.s_measure import SMeasure

from misc import batch_apply_bilateral_solver


@torch.no_grad()
def write_metric_tf(
    writer,
    metrics,
    n_iter = -1,
    name = ""
):
    writer.add_scalar(
        f"Validation/{name}iou_pred",
        metrics["ious"].avg,
        n_iter,
    )
    writer.add_scalar(
        f"Validation/{name}acc_pred",
        metrics["pixel_accs"].avg,
        n_iter,
    )
    writer.add_scalar(
        f"Validation/{name}f_max",
        metrics["f_maxs"].avg,
        n_iter,
    )

@torch.no_grad()
def eval_batch(
    batch_gt_masks, 
    batch_pred_masks,
    metrics_res={},
    reset=False
):
    """
    Evaluation code adapted from SelfMask: https://github.com/NoelShin/selfmask
    """

    f_values = {}
    # Keep track of f_values for each threshold
    for i in range(255):  # should equal n_bins in metrics/f_measure.py
        f_values[i] = AverageMeter()

    if metrics_res == {}:
        metrics_res["f_scores"] = AverageMeter()
        metrics_res["f_maxs"] = AverageMeter()
        metrics_res["f_maxs_fixed"] = AverageMeter()
        metrics_res["f_means"] = AverageMeter()
        metrics_res["maes"] = AverageMeter()
        metrics_res["ious"] = AverageMeter()
        metrics_res["pixel_accs"] = AverageMeter()
        metrics_res["s_measures"] = AverageMeter()

    if reset:
        metrics_res["f_scores"].reset()
        metrics_res["f_maxs"].reset()
        metrics_res["f_maxs_fixed"].reset()
        metrics_res["f_means"].reset()
        metrics_res["maes"].reset()
        metrics_res["ious"].reset()
        metrics_res["pixel_accs"].reset()
        metrics_res["s_measures"].reset()

    # iterate over batch dimension
    for _, (pred_mask, gt_mask) in enumerate(
        zip(batch_pred_masks, batch_gt_masks)
    ):
        assert pred_mask.shape == gt_mask.shape, f"{pred_mask.shape} != {gt_mask.shape}"
        assert len(pred_mask.shape) == len(gt_mask.shape) == 2
        # Compute
        # Binarize at 0.5 for IoU and pixel accuracy
        binary_pred = (pred_mask > 0.5).float().squeeze()
        iou = compute_iou(binary_pred, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)  # soft mask for F measure
        mae = compute_mae(binary_pred, gt_mask)
        pixel_acc = compute_pixel_accuracy(binary_pred, gt_mask)

        # Update
        metrics_res["ious"].update(val=iou.numpy(), n=1)
        metrics_res["f_scores"].update(val=f_measures["f_measure"].numpy(), n=1)
        metrics_res["f_maxs"].update(val=f_measures["f_max"].numpy(), n=1)
        metrics_res["f_means"].update(val=f_measures["f_mean"].numpy(), n=1)
        metrics_res["s_measures"].update(
            val=SMeasure()(pred_mask=pred_mask, gt_mask=gt_mask.to(torch.float32)), n=1
        )
        metrics_res["maes"].update(val=mae.numpy(), n=1)
        metrics_res["pixel_accs"].update(val=pixel_acc.numpy(), n=1)

        # Keep track of f_values for each threshold
        all_f = f_measures["all_f"].numpy()
        for k, v in f_values.items():
            v.update(val=all_f[k], n=1)
        # Then compute the max for the f_max_fixed
        metrics_res["f_maxs_fixed"].update(
            val=np.max([v.avg for v in f_values.values()]), n=1
        )

    results = {}
    # F-measure, F-max, F-mean, MAE, S-measure, IoU, pixel acc.
    results["f_measure"] = metrics_res["f_scores"].avg
    results["f_max"] = metrics_res["f_maxs"].avg
    results["f_maxs_fixed"] = metrics_res["f_maxs_fixed"].avg
    results["f_mean"] = metrics_res["f_means"].avg
    results["s_measure"] = metrics_res["s_measures"].avg
    results["mae"] = metrics_res["maes"].avg
    results["iou"] = float(iou.numpy())
    results["pixel_acc"] = metrics_res["pixel_accs"].avg

    return results, metrics_res

def evaluate_saliency(
    dataset,
    model,
    writer=None,
    batch_size=1,
    n_iter=-1,
    apply_bilateral=False,
    im_fullsize=True,
    method="pred",  # can also be "bkg",
    apply_weights: bool = True,
    evaluation_mode: str = 'single', # choices are ["single", "multi"]
):

    if im_fullsize:
        # Change transformation
        dataset.fullimg_mode()
        batch_size = 1

    valloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    sigmoid = nn.Sigmoid()

    metrics_res = {}
    metrics_res_bs = {}
    valbar = tqdm(enumerate(valloader, 0), leave=None)
    for i, data in valbar:
        inputs, _, gt_labels, _ = data
        inputs = inputs.to("cuda")
        gt_labels = gt_labels.to("cuda").float()

        # Forward step
        with torch.no_grad():
            preds, _, shape_f, att = model.forward_step(
                                        inputs, for_eval=True
                                    )

        if method == "pred":
            h, w = gt_labels.shape[-2:]
            preds_up = F.interpolate(
                preds, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False
            )[..., :h, :w]
            soft_preds = sigmoid(preds_up.detach()).squeeze(0)
            preds_up = (
                (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()
            )

        elif method == "bkg":
            bkg_mask_pred = model.compute_background_batch(
                att, shape_f,
                apply_weights=apply_weights,
            )
            # Transform bkg detection to foreground detection
            obj_mask = (
                ~bkg_mask_pred.bool()
            ).float()  # Obj labels is inverse of bkg

            # Fit predictions to image size
            preds_up = F.interpolate(
                obj_mask.unsqueeze(1),
                gt_labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds_up = (preds_up > 0.5).float()
            soft_preds = preds_up  # not soft actually

        reset = True if i == 0 else False
        if evaluation_mode == 'single':
            labeled, nr_objects = ndimage.label(preds_up.squeeze().cpu().numpy())
            if nr_objects == 0:
                preds_up_one_cc = preds_up.squeeze()
                print("nr_objects == 0")
            else:
                nb_pixel = [np.sum(labeled == i) for i in range(nr_objects + 1)]
                pixel_order = np.argsort(nb_pixel)

                cc = [torch.Tensor(labeled == i) for i in pixel_order]
                cc = torch.stack(cc).cuda()

                # Find CC set as background, here not necessarily the biggest
                cc_background = (
                    (
                        (
                            (~(preds_up[None, :, :, :].bool())).float()
                            + cc[:, None, :, :].cuda()
                        )
                        > 1
                    ).sum(-1).sum(-1).argmax()
                )
                pixel_order = np.delete(
                                pixel_order, int(cc_background.cpu().numpy())
                            )

                preds_up_one_cc = torch.Tensor(labeled == pixel_order[-1]).cuda()

            _, metrics_res = eval_batch(
                gt_labels,
                preds_up_one_cc.unsqueeze(0),
                metrics_res=metrics_res,
                reset=reset,
            )

        elif evaluation_mode == 'multi':
            # Eval without bilateral solver
            _, metrics_res = eval_batch(
                                gt_labels,
                                soft_preds.unsqueeze(0) if len(soft_preds.shape) == 2 else soft_preds,
                                metrics_res=metrics_res,
                                reset=reset,
                            )  # soft preds needed for F beta measure

        # Apply bilateral solver
        preds_bs = None
        if apply_bilateral:
            get_all_cc = True if evaluation_mode == 'multi' else False
            preds_bs, _ = batch_apply_bilateral_solver(data,
                            preds_up.detach(),
                            get_all_cc = get_all_cc
                        )

            _, metrics_res_bs = eval_batch(
                gt_labels,
                preds_bs[None,:,:].float(),
                metrics_res=metrics_res_bs,
                reset=reset
            )

        bar_str = f"{dataset.name} | {evaluation_mode} mode | " \
                  f"F-max {metrics_res['f_maxs'].avg:.3f} " \
                  f"IoU {metrics_res['ious'].avg:.3f}, " \
                  f"PA {metrics_res['pixel_accs'].avg:.3f}"
        
        if apply_bilateral:
            bar_str += f" | with bilateral solver: " \
                       f"F-max {metrics_res_bs['f_maxs'].avg:.3f}, " \
                       f"IoU {metrics_res_bs['ious'].avg:.3f}, " \
                       f"PA. {metrics_res_bs['pixel_accs'].avg:.3f}"

        valbar.set_description(bar_str)

    # Writing in tensorboard
    if writer is not None:
        write_metric_tf(
            writer,
            metrics_res,
            n_iter=n_iter,
            name=f"{dataset.name}_{evaluation_mode}_"
        )
        
        if apply_bilateral:
                write_metric_tf(
                    writer,
                    metrics_res_bs,
                    n_iter=n_iter,
                    name=f"{dataset.name}_{evaluation_mode}-BS_"
                )

    # Go back to original transformation
    if im_fullsize:
        dataset.training_mode()