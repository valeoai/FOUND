"""
Dataset functions for applying Normalized Cut.
Code adapted from SelfMask: https://github.com/NoelShin/selfmask
"""

import os
from typing import Optional, Tuple, Union

from pycocotools.coco import COCO
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from datasets.utils import unnormalize
from datasets.geometric_transforms import resize
from datasets.VOC import get_voc_detection_gt, create_gt_masks_if_voc, create_VOC_loader
from datasets.augmentations import geometric_augmentations, photometric_augmentations

from datasets.uod_datasets import UODDataset

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def set_dataset_dir(dataset_name, root_dir):
    if dataset_name == "ECSSD":
        dataset_dir = os.path.join(root_dir, "ECSSD")
        img_dir = os.path.join(dataset_dir, "images")
        gt_dir = os.path.join(dataset_dir, "ground_truth_mask")

    elif dataset_name == "DUTS-TEST":
        dataset_dir = os.path.join(root_dir, "DUTS")
        img_dir = os.path.join(dataset_dir, "DUTS-TE-Image")
        gt_dir = os.path.join(dataset_dir, "DUTS-TE-Mask")

    elif dataset_name == "DUTS-TR":
        dataset_dir = os.path.join(root_dir, "DUTS")
        img_dir = os.path.join(dataset_dir, "DUTS-TR-Image")
        gt_dir = os.path.join(dataset_dir, "DUTS-TR-Mask")

    elif dataset_name == "DUT-OMRON":
        dataset_dir = os.path.join(root_dir, "DUT-OMRON")
        img_dir = os.path.join(dataset_dir, "DUT-OMRON-image")
        gt_dir = os.path.join(dataset_dir, "pixelwiseGT-new-PNG")

    elif dataset_name == "VOC07":
        dataset_dir = os.path.join(root_dir, "VOC2007")
        img_dir = dataset_dir
        gt_dir = dataset_dir

    elif dataset_name == "VOC12":
        dataset_dir = os.path.join('/datasets_local/osimeoni', "VOC2012")
        img_dir = dataset_dir
        gt_dir = dataset_dir

    elif dataset_name == "COCO17":
        dataset_dir = os.path.join(root_dir, "COCO")
        img_dir = dataset_dir
        gt_dir = dataset_dir

    elif dataset_name == "ImageNet":
        dataset_dir = os.path.join(root_dir, "ImageNet")
        img_dir = dataset_dir
        gt_dir = dataset_dir

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    return img_dir, gt_dir


def build_dataset(
    root_dir: str,
    dataset_name: str,
    dataset_set: Optional[str] = None,
    for_eval: bool = False,
    config=None, 
    evaluation_type="saliency", # uod, 
):
    """
    Build dataset
    """

    if evaluation_type == "saliency":
        img_dir, gt_dir = set_dataset_dir(dataset_name, root_dir)

        dataset = FoundDataset(
            name=dataset_name,
            img_dir=img_dir,
            gt_dir=gt_dir,
            dataset_set=dataset_set,
            config=config,
            for_eval=for_eval,
            evaluation_type=evaluation_type,
        )

    elif evaluation_type == "uod":
        assert dataset_name in ["VOC07", "VOC12", "COCO20k"]
        dataset_set = "trainval" if dataset_name in ["VOC07", "VOC12"] else "train"
        no_hards = False
        dataset = UODDataset(
            dataset_name,
            dataset_set,
            root_dir=root_dir,
            remove_hards=no_hards,
        )

    return dataset


class FoundDataset(Dataset):
    def __init__(
        self,
        name: str,
        img_dir: str,
        gt_dir: str,
        dataset_set: Optional[str] = None,
        config=None,
        for_eval:bool = False,
        evaluation_type:str = "saliency",
    ) -> None:
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.for_eval = for_eval
        self.use_aug =  not for_eval
        self.evaluation_type = evaluation_type

        assert evaluation_type in ["saliency"]

        self.name = name
        self.dataset_set = dataset_set
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        # if VOC dataset
        self.loader = None
        self.cocoGt = None

        self.config = config
        
        if "VOC" in self.name:
            self.loader = create_VOC_loader(self.img_dir, dataset_set, evaluation_type)

        # if ImageNet dataset
        elif "ImageNet" in self.name:
            self.loader = torchvision.datasets.ImageNet(
                self.img_dir,
                split=dataset_set,
                transform=None,
                target_transform=None,
            )

        elif "COCO" in self.name:
            year = int("20"+self.name[-2:])
            annFile=f'/datasets_local/COCO/annotations/instances_{dataset_set}{str(year)}.json'
            self.cocoGt=COCO(annFile)
            self.img_ids = list(sorted(self.cocoGt.getImgIds()))
            self.img_dir = f'/datasets_local/COCO/images/{dataset_set}{str(year)}/'

        # Transformations
        if self.for_eval:
            full_img_transform, no_norm_full_img_transform = self.get_init_transformation(
                isVOC="VOC" in name
            )
            self.full_img_transform = full_img_transform
            self.no_norm_full_img_transform = no_norm_full_img_transform

        # Images
        self.list_images = None
        if not "VOC" in self.name and not "COCO" in self.name:
            self.list_images = [
                os.path.join(img_dir, i) for i in sorted(os.listdir(img_dir))
            ]

        self.ignore_index = -1
        self.mean = NORMALIZE.mean
        self.std = NORMALIZE.std
        self.to_tensor_and_normalize = T.Compose([T.ToTensor(), NORMALIZE])
        self.normalize = NORMALIZE

        if config is not None and self.use_aug:
            self._set_aug(config)


    def get_init_transformation(self, isVOC: bool = False):
        if isVOC:
            t = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float), NORMALIZE])
            t_nonorm = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])
            return t, t_nonorm

        else:
            t = T.Compose([T.ToTensor(), NORMALIZE])
            t_nonorm = T.Compose([T.ToTensor()])
            return t, t_nonorm

    def _set_aug(self, config):
        """
        Set augmentation based on config.
        """
         
        photometric_aug = config.training["photometric_aug"]
        
        self.cropping_strategy = config.training["cropping_strategy"]
        if self.cropping_strategy == "center_crop":
            self.use_aug = False  # default strategy, not considered to be a data aug
        self.scale_range = config.training["scale_range"]
        self.crop_size = config.training["crop_size"]
        self.center_crop_transforms = T.Compose(
            [
                T.CenterCrop((self.crop_size, self.crop_size)),
                T.ToTensor(),
            ]
        )
        self.center_crop_only_transforms = T.Compose(
            [T.CenterCrop((self.crop_size, self.crop_size)), T.PILToTensor()]
        )

        self.proba_photometric_aug = config.training["proba_photometric_aug"]

        self.random_color_jitter = False
        self.random_grayscale = False
        self.random_gaussian_blur = False
        if photometric_aug == "color_jitter":
            self.random_color_jitter = True
        elif photometric_aug == "grayscale":
            self.random_grayscale = True
        elif photometric_aug == "gaussian_blur":
            self.random_gaussian_blur = True

    def _preprocess_data_aug(
        self,
        image: Image.Image,
        mask: Image.Image,
        ignore_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data in a proper form for either training (data augmentation) or validation."""
        
        # resize to base size
        image = resize(
            image,
            size=self.crop_size,
            edge="shorter",
            interpolation="bilinear",
        )
        mask = resize(
            mask,
            size=self.crop_size,
            edge="shorter",
            interpolation="bilinear",
        )

        if not isinstance(mask, torch.Tensor):
            mask: torch.Tensor = torch.tensor(np.array(mask))

        random_scale_range = None
        random_crop_size = None
        random_hflip_p = None
        if self.cropping_strategy == "random_scale":
            random_scale_range = self.scale_range
        elif self.cropping_strategy == "random_crop":
            random_crop_size = self.crop_size
        elif self.cropping_strategy == "random_hflip":
            random_hflip_p = 0.5
        elif self.cropping_strategy == "random_crop_and_hflip":
            random_hflip_p = 0.5
            random_crop_size = self.crop_size

        if random_crop_size or random_hflip_p or random_scale_range:
            image, mask = geometric_augmentations(
                image=image,
                mask=mask,
                random_scale_range=random_scale_range,
                random_crop_size=random_crop_size,
                ignore_index=ignore_index,
                random_hflip_p=random_hflip_p,
            )

        if random_scale_range:
            # resize to (self.crop_size, self.crop_size)
            image = resize(
                image,
                size=self.crop_size,
                interpolation="bilinear",
            )
            mask = resize(
                mask,
                size=(self.crop_size, self.crop_size),
                interpolation="bilinear",
            ) 

        image = photometric_augmentations(
            image,
            random_color_jitter=self.random_color_jitter,
            random_grayscale=self.random_grayscale,
            random_gaussian_blur=self.random_gaussian_blur,
            proba_photometric_aug=self.proba_photometric_aug,
        )

        # to tensor + normalize image
        image = self.to_tensor_and_normalize(image)

        return image, mask

    def __len__(self) -> int:
        if "VOC" in self.name:
            return len(self.loader)
        elif "ImageNet" in self.name:
            return len(self.loader)
        elif "COCO" in self.name:
            return len(self.img_ids)
        return len(self.list_images)

    def _apply_center_crop(
        self, image: Image.Image, mask: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = self.center_crop_transforms(image)
        # need to normalize image
        img_t = self.normalize(img_t)
        mask_gt = self.center_crop_transforms(mask).squeeze()
        return img_t, mask_gt


    def __getitem__(self, idx, get_mask_gt=True):
        if "VOC" in self.name:
            img, gt_labels = self.loader[idx]
            if self.evaluation_type == "uod":
                gt_labels, _ = get_voc_detection_gt(
                    gt_labels, remove_hards=False
                )
            elif self.evaluation_type == "saliency":
                mask_gt = create_gt_masks_if_voc(gt_labels)
            img_path = self.loader.images[idx]

        elif "ImageNet" in self.name:
            img, _ = self.loader[idx]
            img_path = self.loader.imgs[idx][0]
            # empty mask since no gt mask, only class label
            zeros = np.zeros(np.array(img).shape[:2])
            mask_gt = Image.fromarray(zeros)

        elif "COCO" in self.name:
            img_id = self.img_ids[idx]

            path = self.cocoGt.loadImgs(img_id)[0]["file_name"]
            img =  Image.open(os.path.join(self.img_dir, path)).convert("RGB")
            _ = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(id))
            img_path = self.img_ids[idx] # What matters most is the id for eval

            # empty mask since no gt mask, only class label
            zeros = np.zeros(np.array(img).shape[:2])
            mask_gt = Image.fromarray(zeros)

        # For all others
        else:
            img_path = self.list_images[idx]
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
                im_name = img_path.split("/")[-1]
                mask_gt = Image.open(
                    os.path.join(self.gt_dir, im_name.replace(".jpg", ".png"))
                ).convert("L")

        if self.for_eval:
            img_t = self.full_img_transform(img)
            img_init = self.no_norm_full_img_transform(img)

            if self.evaluation_type == "saliency":
                mask_gt = torch.tensor(np.array(mask_gt)).squeeze()
                mask_gt = np.array(mask_gt)
                mask_gt = mask_gt == 255
                mask_gt = torch.tensor(mask_gt)
        else:
            if self.use_aug:
                img_t, mask_gt = self._preprocess_data_aug(
                    image=img, mask=mask_gt, ignore_index=self.ignore_index
                )
                mask_gt = np.array(mask_gt)
                mask_gt = mask_gt == 255
                mask_gt = torch.tensor(mask_gt)
            else:
                # no data aug
                img_t, mask_gt = self._apply_center_crop(image=img, mask=mask_gt)
                gt_labels = self.center_crop_only_transforms(gt_labels).squeeze()
                mask_gt = np.asarray(mask_gt, np.int64)
                mask_gt = mask_gt == 1
                mask_gt = torch.tensor(mask_gt)

            img_init = unnormalize(img_t)

        if not get_mask_gt:
            mask_gt = None

        if self.evaluation_type == "uod":
            gt_labels = torch.tensor(gt_labels)
            mask_gt = gt_labels

        return img_t, img_init, mask_gt, img_path

    def fullimg_mode(self):
        self.val_full_image = True

    def training_mode(self):
        self.val_full_image = False