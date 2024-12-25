import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd

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
