import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class GaussianBlur:
    """
    Code borrowed from SelfMask: https://github.com/NoelShin/selfmask
    """

    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size: float, min: float = 0.1, max: float = 2.0) -> None:
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample: Image.Image, random_gaussian_blur_p: float):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            import cv2

            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(
                sample, (self.kernel_size, self.kernel_size), sigma
            )
        return sample


def unnormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Code borrowed from STEGO: https://github.com/mhamilton723/STEGO
    """
    image2 = torch.clone(image)
    for t, m, s in zip(image2, mean, std):
        t.mul_(s).add_(m)

    return image2
