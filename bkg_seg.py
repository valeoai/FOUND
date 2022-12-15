# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
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
import torch.nn.functional as F

from typing import Tuple

def compute_img_bkg_seg(
    attentions,
    feats,
    featmap_dims,
    th_bkg,
    dim=64,
    epsilon: float = 1e-10,
    apply_weights: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    inputs
       - attentions [B, ]
    """
    
    w_featmap, h_featmap = featmap_dims

    nb, nh, _ = attentions.shape[:3]
    # we keep only the output patch attention
    att = attentions[:, :, 0, 1:].reshape(nb, nh, -1)
    att = att.reshape(nb, nh, w_featmap, h_featmap)

    # -----------------------------------------------
    # Inspired by CroW sparsity channel weighting of each head CroW, Kalantidis etal.
    threshold = torch.mean(att.reshape(nb, -1), dim=1)  # Find threshold per image
    Q = torch.sum(
        att.reshape(nb, nh, w_featmap * h_featmap) > threshold[:, None, None], axis=2
    ) / (w_featmap * h_featmap)
    beta = torch.log(torch.sum(Q + epsilon, dim=1)[:, None] / (Q + epsilon))

    # Weight features based on attention sparsity
    descs = feats[:,1:,]
    if apply_weights:
        descs = (descs.reshape(nb, -1, nh, dim) * beta[:, None, :, None]).reshape(
            nb, -1, nh * dim
        )
    else:
        descs = (descs.reshape(nb, -1, nh, dim)).reshape(
            nb, -1, nh * dim
        )

    # -----------------------------------------------
    # Compute cosine-similarities
    descs = F.normalize(descs, dim=-1, p=2)
    cos_sim = torch.bmm(descs, descs.permute(0, 2, 1))

    # -----------------------------------------------
    # Find pixel with least amount of attention
    if apply_weights:
        att = att.reshape(nb, nh, w_featmap, h_featmap) * beta[:, :, None, None]
    else:
        att = att.reshape(nb, nh, w_featmap, h_featmap) 
    id_pixel_ref = torch.argmin(torch.sum(att, axis=1).reshape(nb, -1), dim=-1)

    # -----------------------------------------------
    # Mask of definitely background pixels: 1 on the background
    cos_sim = cos_sim.reshape(nb, -1, w_featmap * h_featmap)

    bkg_mask = (
        cos_sim[torch.arange(cos_sim.size(0)), id_pixel_ref, :].reshape(
            nb, w_featmap, h_featmap
        )
        > th_bkg
    )  # mask to be used to remove background

    return bkg_mask.float()