import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

#The neural network architecture as defined in the dl_approach.py file
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

# Load the model 
model = CrowdNet()
model.load_state_dict(torch.load('models/best_crowdnet.pth', map_location=torch.device('cpu')))
model.eval()

def predict_and_overlay(image, flip_colors = False, overlay_strength = 0.6, output_path="overlay.png"):
    """
    This function uses the CrowdNet model and predicts the density map for the input image.
    It then overlays the density map on the input image and saves the output image.
    """
    # Resize the image, since the model is trained on a 1920x1080 image
    orig = np.array(image)
    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    orig = cv2.resize(orig, (1920, 1080))

    #Preprocess the image
    tensor_image = torch.from_numpy(orig).permute(2, 0, 1).float() / 255.0
    tensor_image = tensor_image.unsqueeze(0)

    # Predict the density map
    with torch.no_grad():
        output = model(tensor_image)[0][0].cpu().numpy()

    # Normalize the heatmap output 
    heatmap = (output - output.min()) / (output.max() - output.min() + 1e-8) #Extremely small value added to prevent divinding by 0
    heatmap = (heatmap * 255).astype(np.uint8)

    # If the flip colors heatmap is selected, we will flip the colors
    if flip_colors:
        heatmap = 255 - heatmap

    #Apply the color mapping
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    #Overlay the heatmap on the original image, using the preset strength
    alpha = 1 - overlay_strength
    beta = overlay_strength
    overlay = cv2.addWeighted(orig, alpha, heatmap_color, beta, 0)

    # Convert the overlay to allow for Streamlit to display it
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Save the overlay image to a BytesIO object
    buffer = BytesIO()
    Image.fromarray(overlay_rgb).save(buffer, format="PNG")
    buffer.seek(0)

    return buffer

# Streamlit app UI
st.title("CrowdNet Density Map Overlay")
st.write("Upload an image to see the density map overlay.")
st.write("For best results, please use images from a birds-eye view, as well as a mixture of crowded and sparse sections")
st.write("You can also flip the colors of the heatmap or set the overlay strength to make it more visible")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    #Show the image uploaded
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Uploaded Image", use_container_width=True)

    # A slider for the overlay stength
    overlay_strength = st.slider("Overlay Strength", 0.1, 1.0, 0.6, step=0.05)

    # A checkbox for flipping the colors
    flip_colors = st.checkbox("Flip Colors", value=False)

    # When the button is pressed, we gen display the overlay
    if st.button("Generate Overlay"):
        # Predict and overlay the density map
        overlay_buffer = predict_and_overlay(original_image, flip_colors, overlay_strength)

        # Display the overlay image
        st.image(overlay_buffer, caption="Overlay Image", use_container_width=True)

        #Allow the user to download the overlay image if they want to
        st.download_button(
            label="Download Overlay Image",
            data=overlay_buffer,
            file_name="overlay.png",
            mime="image/png"
        )
        st.success("Overlay generated successfully!")
