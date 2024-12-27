import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as func

def landmarks_to_mask(image, landmarks):
    """
    Convert a list of landmark points on an image into a binary mask for the model
    Args:
        image: Path to image, assuming is in the same root directory
        landmarks (list): List of landmark points representing the outline of the stomata, 
                          ex. [(x1, y1), (x2, y2), ..., (xn, yn)].
    Returns:
        torch.Tensor: Binary mask of the stomata.
    """
    image_size = Image.open(image).size[::-1]  # to height, width for numpy
    mask = np.zeros(image_size, dtype=np.uint8)

    landmarks = np.array(landmarks, dtype=np.int32)

    cv2.fillPoly(mask, [landmarks], 1)

    return torch.tensor(mask, dtype=torch.float32)

# Example use
# mask = landmarks_to_mask(image, landmarks)
# plt.imshow(mask, cmap='gray')
# plt.show()

class PadToDivisible:
    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:  # [B, C, H, W]
                _, _, h, w = image.shape
            elif len(image.shape) == 3:  # [C, H, W]
                _, h, w = image.shape
            else:
                raise ValueError(f"Unexpected tensor shape: {image.shape}")
        else:  # Assuming PIL image
            w, h = image.size  

        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor

        if isinstance(image, torch.Tensor):
            return func.pad(image, (0, pad_w, 0, pad_h))  
        else:
            from PIL import ImageOps
            return ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
