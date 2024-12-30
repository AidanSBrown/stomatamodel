import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as func
import geopandas as gpd
import pandas as pd
import shapely
import os
from pathlib import Path

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def landmarks_to_mask(image_path, landmarks_dict):
    """
    Convert a list of landmark points on an image into a binary mask for the model
    Args:
        image_path: Path to image, assuming is in the same root directory
        landmarks_dict: Dictionary of landmarks grouped by image from csvcoordstogroup
    Returns:
        torch.Tensor: Binary mask of the stomata.
    """
    img_name = os.path.basename(image_path)

    if img_name not in landmarks_dict:
        raise ValueError(f"No landmarks found for image {img_name}")

    image_size = Image.open(image_path).size[::-1] # W/H to H/W
    mask = np.zeros(image_size, dtype=np.uint8)

    landmarks_list = landmarks_dict[img_name]
    
    for landmarks in landmarks_list:
        landmarks = np.array(landmarks, dtype=np.int32)
        cv2.fillPoly(mask, [landmarks], 1)

    return torch.tensor(mask, dtype=torch.float32)

def csvcoordstogroup(csv_path):
    """Group image stomata"""

    df = pd.read_csv(csv_path)
    image_landmarks = {}
    
    for image_name, group in df.groupby('image_name'):
        landmarks_list = []
        
        for _, row in group.iterrows():
            landmarks = []
            for i in range(20): 
                x = row[f'x{i+1}']
                y = row[f'y{i+1}']
                landmarks.append((x, y))
            landmarks_list.append(landmarks)
            
        image_landmarks[image_name] = landmarks_list
    
    return image_landmarks

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

def shptocsv(shapefile_path, image_path, outpath, flipyaxis = True):
    """
    Convert shapefile of annotations 
    """
    gdf = gpd.read_file(shapefile_path)
    
    rows = []
    
    for idx, geom in enumerate(gdf.geometry):
        if geom.is_empty or geom.geom_type != "Polygon":
            continue
        
        coords = list(geom.exterior.coords)
        
        coords = coords[:20]
        
        flattened = [coord for point in coords for coord in point]
        row = flattened + [None] * (2 * 20 - len(flattened))   
        rows.append(row)

    columns = [f"{xy}{i}" for i in range(1, 21) for xy in ("x", "y")]
    
    df = pd.DataFrame(rows, columns=columns)
    df.dropna(how='all')
    
    if flipyaxis: # QGIS y axis flipped for images without geographic data
        y_columns = [f'y{i}' for i in range(1, 21)]
        try:
            for col in y_columns:
                df[col] = df[col].apply(lambda x: -x if x < 0 else x)
        except TypeError as e:
            pass

    df.insert(0, "image_name", os.path.basename(image_path))
    df.to_csv(outpath, index=False,na_rep="") # na rep supposed to exclude trailing commas but doesn't work

# shptocsv('/Users/aidanbrown/Desktop/brownsville/stomata_train_13.shp','/Users/aidanbrown/Desktop/brownsville/BRO_ILEOPA_Train13.tif','data/train13.csv')


def visualize_mask(image, mask, device="cpu"):
    """
    Visualizes the images and masks from dataloader to ensure accuracy

    NOTE: Set batch size to one tp avoid error from invalid shape
    """  
    image.to(device)  
    mask.to(device)  
    
    
    if image.ndim == 4: # If four channels ex [16, 3, 512, 512] Then take first img
        image = image[0]
    if mask.ndim == 4:  
        mask = mask[0]

    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().cpu().numpy() 
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.axis("off")
    plt.title("Input Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")
    plt.title("Ground Truth Mask")
        
    plt.tight_layout()
    plt.show()
