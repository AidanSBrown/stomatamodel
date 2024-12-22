import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F

# Set GPU or CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")

# Initialize model
class NerualNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)  # 3 channels (RGB)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.center = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final output layer (Binary mask)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels): # Conv block is for repeatability when having multiple convolutionary layers
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        center = self.center(F.max_pool2d(enc4, 2))

        # Decoder path
        dec4 = self.dec4(F.upsample(center, scale_factor=2, mode='bilinear', align_corners=True))
        dec4 = dec4 + enc4  # Skip connection

        dec3 = self.dec3(F.upsample(dec4, scale_factor=2, mode='bilinear', align_corners=True))
        dec3 = dec3 + enc3

        dec2 = self.dec2(F.upsample(dec3, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = dec2 + enc2

        dec1 = self.dec1(F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = dec1 + enc1

        output = torch.sigmoid(self.final(dec1))
        return output

model = NerualNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device) # Creates a tensor of random image / data


# stomata_image = torch.tensor(data)


params = list(model.parameters())
print(len(params))
print(params[0].size())  

