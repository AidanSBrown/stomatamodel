import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

points_frame = pd.read_csv('data/faces/face_landmarks.csv')


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

for n, row in points_frame.iterrows():
    img_name = points_frame.iloc[n, 0]
    landmarks = points_frame.iloc[n, 1:]
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    plt.figure()
    show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
    plt.show()