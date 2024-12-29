import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as func 
import torch.nn.functional as F
import torch.optim as optim
from dataloader import StomataDataset, data_transform
import matplotlib.pyplot as plt
from PIL import Image
from utils import landmarks_to_mask, PadToDivisible
import time

# Set GPU or CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")

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

        # Binary mask
        self.final = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
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

        dec4 = self.dec4(F.interpolate(center, scale_factor=2, mode='nearest'))
        dec4 = dec4 + enc4 

        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2, mode='nearest'))
        dec3 = dec3 + enc3
        
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='nearest'))
        dec2 = dec2 + enc2
        
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='nearest'))
        dec1 = dec1 + enc1
   
        output = torch.sigmoid(self.final(dec1))
        return output

#                            Reading block                           #
# print(model)
# params = list(model.parameters())
# print(len(params))
# print(params[0].size())  

def train(model,train_csv,device,epochs=5, batch_size=16):
    train_set = StomataDataset(csv_file=train_csv,
                                    root_dir=os.path.dirname(train_csv),
                                    transform = data_transform)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0) # In the past my machine has been bad with multiple workers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for image, labels in trainloader:
            input, target = image.to(device), labels.to(device)

            target = target.unsqueeze(0) # To resolve tensor size mismatch
            target = target.permute(1, 0, 2, 3)  

            output = model(input) # Forward pass
        
            loss = loss_fn(output, target)
            
            optimizer.zero_grad()  # Backwards pass
            loss.backward()      
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    print(f"Traning complete in {time.time() - start_time}")
    return model

#### Example training use ####
# model = Stomatann().to(device)
# model = train(model,"data/train1.csv",device,batch_size=8,epochs=5)
# torch.save(model.state_dict(), "models/stomatamodel_v1.pth")

def predict(model_path, image_path=str, device="cpu", show=True, image_size=512):
    """
    Predict on an image 
    args: 
        model: Path to pth model to load and make predictions
        image: Path to image to predict on
        device: Default cpu 
    """
    model = Stomatann().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path)
    # try: # Should check if this is really necessary        # In case need of resizing?
    #     image = F.resize(image, [image_size, image_size])
    # except RuntimeError as e:
    #     print(f"Target size of size {image_size} too large, changing target size")
    #     image.thumbnail((image_size,image_size), Image.Resampling.LANCZOS)
    image = func.pil_to_tensor(image).float()
    image = image.to(device).unsqueeze(0) 
    transform = transforms.Compose([
        PadToDivisible(divisor=32),]) # To prevent tensor size mismatch error
    image = transform(image)

    with torch.no_grad():
        output = model(image)

    # Sigmoid fn to get probabilities and then threshold to get binary mask
    mask = torch.sigmoid(output) > 0.5
    if show:
        print("Showing image prediction")
        mask = mask.numpy().squeeze(0).squeeze(0)
        image = image.numpy().squeeze().squeeze()
        if image.ndim == 3:  # [C, H, W]
            image = image.transpose(1, 2, 0)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='viridis')
        plt.colorbar(label='Mask Intensity')
        plt.axis('off')
        plt.title('Predicted stomata')
        plt.show()

    else:
        return mask.squeeze(0)  # Remove batch dimension

#### Example Predict Use ####
# predict(model_path='models/stomatamodel_v1.pth',
#         image_path = 'data/BRO_PINTAE_Train5.png')