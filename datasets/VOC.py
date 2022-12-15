import os
from typing import Optional, Tuple, Union, Dict, List

import cv2
from pycocotools.coco import COCO
import numpy as np
import torch
import torchvision
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
from tqdm import tqdm

VOCDetectionMetadataType = Dict[str, Dict[str, Union[str, Dict[str, str], List[str]]]]

def get_voc_detection_gt(
    metadata: VOCDetectionMetadataType, remove_hards: bool = False
) -> Tuple[np.array, List[str]]:
    objects = metadata["annotation"]["object"]
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for object in range(nb_obj):
        if remove_hards and (
            objects[object]["truncated"] == "1"
            or objects[object]["difficult"] == "1"
        ):
            continue

        gt_cls = objects[object]["name"]
        gt_clss.append(gt_cls)
        obj = objects[object]["bndbox"]
        x1y1x2y2 = [
            int(obj["xmin"]),
            int(obj["ymin"]),
            int(obj["xmax"]),
            int(obj["ymax"]),
        ]

        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        x1y1x2y2[0] -= 1
        x1y1x2y2[1] -= 1
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss

def create_gt_masks_if_voc(labels: PngImagePlugin.PngImageFile) -> Image.Image:
    mask = np.array(labels)
    mask_gt = (mask > 0).astype(float)
    mask_gt = np.where(mask_gt != 0.0, 255, mask_gt)
    mask_gt = Image.fromarray(np.uint8(mask_gt))
    return mask_gt

def create_VOC_loader(img_dir, dataset_set, evaluation_type):
    year = img_dir[-4:]
    download = not os.path.exists(img_dir)
    if evaluation_type == "uod":
        loader = torchvision.datasets.VOCDetection(
            img_dir,
            year=year,
            image_set=dataset_set,
            transform=None,
            download=download,
        )
    elif evaluation_type == "saliency":
        loader = torchvision.datasets.VOCSegmentation(
            img_dir,
            year=year,
            image_set=dataset_set,
            transform=None,
            download=download,
        )
    else:
        raise ValueError(f"Not implemented for {evaluation_type}.")
    return loader