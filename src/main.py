import os
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from dataloader import StomataDataset, data_transform

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
class Stomatann(nn.Module):
    def __init__(self):
        super(Stomatann,self).__init__()
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

        center = self.center(F.max_pool2d(enc4, 2))

        dec4 = self.dec4(F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=True))
        dec4 = dec4 + enc4 

        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True))
        dec3 = dec3 + enc3
        
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = dec2 + enc2
        
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = dec1 + enc1
   
        output = torch.sigmoid(self.final(dec1))
        return output

#                            Reading block                           #
# print(model)
# params = list(model.parameters())
# print(len(params))
# print(params[0].size())  

# input = torch.randn(1, 3, 256, 256)  # Example input
# target = torch.randn(1, 1, 256, 256)  # Example target
# input = input.to(device)
# target = target.to(device)

# output = model(input) # Forward pass
# loss_fn = nn.BCELoss()

# loss = loss_fn(output, target)
# print(loss)

# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer.zero_grad()  
# loss.backward()      
# optimizer.step()

def train(model,train_csv,device,epochs=10, batch_size=16):
    train_set = StomataDataset(csv_file=train_csv,
                                    root_dir=os.path.dirname(train_csv),
                                    transform = data_transform)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0) # In the past my machine has been bad with multiple workers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for image, labels in trainloader:
            input, target = image.to(device), labels.to(device)
            output = model(input) # Forward pass
        
            loss = loss_fn(output, target)
            
            optimizer.zero_grad()  # Backwards pass
            loss.backward()      
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    print("Traning complete")
    return model

#### Example use ####

model = Stomatann().to(device)

model = train(model,"data/faces/face_landmarks_train.csv",device)
torch.save(model, "models/testingmodel.pth")

def predict(model, image, device):
    model.eval()
    image = image.to(device).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)

    # Sigmoid fn to get probabilities and then threshold to get binary mask
    mask = torch.sigmoid(output) > 0.5
    return mask.squeeze(0)  # Remove batch dimension
