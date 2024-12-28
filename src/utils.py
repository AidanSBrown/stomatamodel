import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as func
import geopandas as gpd
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

def shptocsv(shapefile_path, outpath):
    """
    Convert shapefile of annotations 
    """

    df = gpd.read_file(shapefile_path)
    print(df.head())

    if not df.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("The shapefile contains unsupported geometry types. Expected multipolygon")

    columns = [f"x{i}" if i % 2 != 0 else f"y{i // 2}" for i in range(1, 20 * 2 + 1)]
    df[columns] = pd.DataFrame(df.geometry.apply(lambda geom: extract_points(geom)).tolist(), index=df.index)

    # Save to CSV
    columns_to_save = columns + [col for col in df.columns if col not in ["geometry"]]
    df[columns_to_save].to_csv(outpath, index=False)

def extract_points(geometry, max_points=20):
    if geometry.is_empty or geometry is None:
        return [None] * (max_points * 2)
    
    # Extract coordinates from the exterior ring of the polygon
    coords = list(geometry.exterior.coords)[:max_points]
    flat_coords = [coord for point in coords for coord in point]  # Flatten to x1, y1, x2, y2, ...
    
    # Pad with None if fewer than max_points
    return flat_coords + [None] * (max_points * 2 - len(flat_coords))

shptocsv('/Users/aidanbrown/Desktop/brownsville/stomata_train_01.shp','data/testcsv.csv')


