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

import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from model import FoundModel
from misc import load_config
from torchvision import transforms as T

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Evaluation of FOUND',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--img-path", type=str, default="data/examples/VOC07_000007.jpg", help="Image path."
    )
    parser.add_argument(
        "--model-weights", type=str, default="data/weights/decoder_weights.pt",
    )
    parser.add_argument(
        "--config", type=str, default="configs/found_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
    )
    args = parser.parse_args()

    # Saving dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Configuration
    config = load_config(args.config)

    # ------------------------------------
    # Load the model
    model = FoundModel(vit_model=config.model["pre_training"],
                        vit_arch=config.model["arch"],
                        vit_patch_size=config.model["patch_size"],
                        enc_type_feats=config.found["feats"],
                        bkg_type_feats=config.found["feats"],
                        bkg_th=config.found["bkg_th"])
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # Load the image
    with open(args.img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

        t = T.Compose([T.ToTensor(), NORMALIZE])
        img_t = t(img)[None,:,:,:]
        inputs = img_t.to("cuda")
    
    # Forward step
    with torch.no_grad():
        preds, _, shape_f, att = model.forward_step(inputs, for_eval=True)

    # Apply FOUND
    sigmoid = nn.Sigmoid()
    h, w = img_t.shape[-2:]
    preds_up = F.interpolate(
        preds, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False
    )[..., :h, :w]
    preds_up = (
        (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()
    )

    plt.figure()
    plt.imshow(img)
    plt.imshow(preds_up.cpu().squeeze().numpy(), 'gray', interpolation='none', alpha=0.5)
    plt.axis('off')
    img_name = args.img_path
    img_name = img_name.split('/')[-1].split('.')[0]
    plt.savefig(os.path.join(args.output_dir, f'{img_name}-found.png'), bbox_inches='tight', pad_inches=0)
    plt.close()