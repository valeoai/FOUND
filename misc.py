import os
import random

import re
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from bilateral_solver import bilateral_solver_output


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def load_config(config_file):
    with open(config_file, errors='ignore') as f:
        # conf = yaml.safe_load(f)  # load config
        conf = yaml.load(f, Loader=loader)
    print('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in conf.items()))
        
    #TODO yaml_save(save_dir / 'config.yaml', conf)
    return Struct(**conf)

def set_seed(seed: int) -> None:
    """
    Set all seeds to make results reproducible
    """
    # env
    os.environ["PYTHONHASHSEED"] = str(seed)

    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def IoU(mask1, mask2):
    """
    Code adapted from TokenCut: https://github.com/YangtaoWANG95/TokenCut
    """
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()

def batch_apply_bilateral_solver(data,
                                 masks,
                                 get_all_cc=True,
                                 shape=None):

    cnt_bs = 0
    masks_bs = []
    inputs, init_imgs, gt_labels, img_path = data

    for id in range(inputs.shape[0]):
        _, bs_mask, use_bs = apply_bilateral_solver(
            mask=masks[id].squeeze().cpu().numpy(),
            img=init_imgs[id],
            img_path=img_path[id],
            im_fullsize=False,
            # Careful shape should be opposed
            shape=(gt_labels.shape[-1], gt_labels.shape[-2]),
            get_all_cc=get_all_cc,
        )
        cnt_bs += use_bs

        # use the bilateral solver output if IoU > 0.5
        if use_bs:
            if shape is None:
                shape = masks.shape[-2:]
            # Interpolate to downsample the mask back
            bs_ds = F.interpolate(
                torch.Tensor(bs_mask).unsqueeze(0).unsqueeze(0),
                shape,  # TODO check here
                mode="bilinear",
                align_corners=False,
            )
            masks_bs.append(bs_ds.bool().cuda().squeeze()[None, :, :])
        else:
            # Use initial mask
            masks_bs.append(masks[id].cuda().squeeze()[None, :, :])
    
    return torch.cat(masks_bs).squeeze(), cnt_bs


def apply_bilateral_solver(
    mask,
    img,
    img_path,
    shape,
    im_fullsize=False,
    get_all_cc=False,
    bs_iou_threshold: float = 0.5,
    reshape: bool = True,
):
    # Get initial image in the case of using full image
    img_init = None
    if not im_fullsize:
        # Use the image given by dataloader
        shape = (img.shape[-1], img.shape[-2])
        t = T.ToPILImage()
        img_init = t(img)

    if reshape:
        # Resize predictions to image size
        resized_mask = cv2.resize(mask, shape)
        sel_obj_mask = resized_mask
    else:
        resized_mask = mask
        sel_obj_mask = mask

    # Apply bilinear solver
    _, binary_solver = bilateral_solver_output(
        img_path,
        resized_mask,
        img=img_init,
        sigma_spatial=16,
        sigma_luma=16,
        sigma_chroma=8,
        get_all_cc=get_all_cc,
    )

    mask1 = torch.from_numpy(resized_mask).cuda()
    mask2 = torch.from_numpy(binary_solver).cuda().float()

    use_bs = 0
    # If enough overlap, use BS output
    if IoU(mask1, mask2) > bs_iou_threshold:
        sel_obj_mask = binary_solver.astype(float)
        use_bs = 1

    return resized_mask, sel_obj_mask, use_bs