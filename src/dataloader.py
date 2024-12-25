import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


class StomataDataset(Dataset):
    """Stomata Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file) #data/faces/face_landmarks.csv
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        landmarks = self.annotations.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            image = self.transform(image)
        # sample = {'image': image, 'landmarks': landmarks}

        return image, landmarks

data_transform = transforms.Compose([
        transforms.Resize((100,100)), # Need to change when applying to image j images since they are different sizes and proportions
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_set = StomataDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/',
                                    transform = data_transform)

test_set = StomataDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/',
                                    transform = data_transform)

# trainloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0) # In the past my machine has been bad with multiple workers
# testloader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

# Test/debug
# for image, labels in trainloader:
#     print(image.shape, labels.shape) 
#     break


# for i, sample in enumerate(dataset):
#     print(i, sample['image'].shape, sample['landmarks'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)

#     if i == 3:
#         plt.show()
#         break




