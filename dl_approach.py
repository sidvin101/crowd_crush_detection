from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torchvision.models as models
import glob
import os

# A Dataset class callsed DensityDataset
class DensityDataset(Dataset):
    def __init__(self, image_dir, density_dir, transform=None):
        #Declare the folder paths for the image and density maps
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.density_paths = sorted(glob.glob(os.path.join(density_dir, '*.png')))
        self.transform = transform

    #size of the dataset
    def __len__(self):
        return len(self.image_paths)
    
    # Get the item and density map
    def __getitem__(self, idx):

        #Image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1920, 1080)) # Because of how I trained the model, I had it so that the image is set to this size. It might lead to grainy results, but it should work

        #Density map
        heatmap = cv2.imread(self.density_paths[idx], cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, (1920, 1080))

        #Normalize the image and density map
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float() / 255.0

        return img, heatmap
    
# This class will be our neural network
class CrowdNet(nn.Module):
    def __init__(self):
        super(CrowdNet, self).__init__()

        #We will use a pretrained ResNet18 model, since it is very lightweight
        base = models.resnet18(pretrained=True) 

        #Our model will have two parts, an encoder and a decoder
        # The encoder will be the ResNet18 model, but only the first few layers
        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2
        )

        # The decoder will be a few convolutional layers that will upsample the feature maps
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=1, padding=1),
            nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=False) # this is the size of the original image
        )

        def forward(self, x):
            #Encoder
            x = self.encoder(x)
            #Decoder
            x = self.decoder(x)
            return x
    
def train_model(train_loader, val_loader):
    """
    This function will train the crowdnet model.
    """

    # Instianate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowdNet().to(device)

    # We will use the MSE as our loss function
    criterion = nn.MSELoss()

    # We will use Adam as our optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # We will use 10 epochs for training
    num_epochs = 10

    #In case of any model crashes or early stoppage, this variable will save the best performing model
    best_val_loss = float('inf')

    #Model training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, heatmaps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # Grabs the training MSE loss
        train_loss /= len(train_loader)

        #Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, heatmaps in tqdm(val_loader, desc="Validating", unit="batch"):
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        #At the end of the epoch, we will check and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_crowdnet.pth")
            print("Best model saved!")

def test_model(test_loader):
    """
    This function will test the crowdnet model.
    """
    #Creates the dataset and dataloaders
    test_dataset = DensityDataset("Dataset/test/images", "Dataset/test/heatmaps")
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Instianate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowdNet().to(device)

    # Load the best model
    model.load_state_dict(torch.load("models/best_crowdnet.pth"))
    model.eval()

    # We will use the MSE as our loss function
    criterion = nn.MSELoss()

    #Testing
    test_loss = 0.0

    with torch.no_grad():
        for images, heatmaps in tqdm(test_loader, desc="Testing", unit="batch"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            test_loss += loss.item()

if __name__ == "__main__":
    #Creates the dataset and dataloaders
    train_dataset = DensityDataset("Dataset/train/images", "Dataset/train/heatmaps")
    val_dataset   = DensityDataset("Dataset/val/images", "Dataset/val/heatmaps")
    test_dataset  = DensityDataset("Dataset/test/images", "Dataset/test/heatmaps")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_model(train_loader, val_loader)
    print("Model trained!")

    test_model(test_loader)
    print("Model tested!")
