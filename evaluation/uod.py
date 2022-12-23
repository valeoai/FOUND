# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
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

"""
Code adapted from previous method LOST: https://github.com/valeoai/LOST
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from misc import bbox_iou, get_bbox_from_segmentation_labels


def evaluation_unsupervised_object_discovery(
    dataset,
    model,
    evaluation_mode: str = 'single', # choices are ["single", "multi"]
    output_dir:str = "outputs",
    no_hards:bool = False,
):
    
    assert evaluation_mode == "single"

    sigmoid = nn.Sigmoid()

    # ----------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    start_time = time.time()
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / model.vit_patch_size) * model.vit_patch_size),
            int(np.ceil(img.shape[2] / model.vit_patch_size) * model.vit_patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # # Move to gpu
        img = img.cuda(non_blocking=True)
        
        # Size for transformers
        w_featmap = img.shape[-2] // model.vit_patch_size
        h_featmap = img.shape[-1] // model.vit_patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

        if gt_bbxs is not None:
            # Discard images with no gt annotations
            # Happens only in the case of VOC07 and VOC12
            if gt_bbxs.shape[0] == 0 and no_hards:
                continue

        outputs = model.forward_step(img[None, :, :, :])
        preds = (sigmoid(outputs[0].detach()) > 0.5).float().squeeze().cpu().numpy()

        # get bbox
        pred = get_bbox_from_segmentation_labels(
            segmenter_predictions=preds,
            scales=[model.vit_patch_size, model.vit_patch_size],
            initial_image_size=init_image_size[1:],
        )

        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred


        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
       
        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    # Evaluate
    print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
    result_file = os.path.join(output_dir, 'uod_results.txt')
    with open(result_file, 'w') as f:
        f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
    print('File saved at %s'%result_file)