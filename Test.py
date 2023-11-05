import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from UnetModel import *

# Load the pre-trained U-Net model
model = Unet(n_classes=21)  # Create an instance of your U-Net model class
model.load_state_dict(torch.load("modelUNet_ep_70.pth", map_location=torch.device('cpu')))  # Use 'cuda' if available
model.eval()

# Load and preprocess the input image
image = Image.open("img/2007_000170.jpg")  # Replace with the path to your image
transform = transforms.Compose([transforms.Resize((256, 256)),  # Resize to match the model's input size
                               transforms.ToTensor(),  # Convert to a PyTorch tensor
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # Normalize as needed
input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

# Move the model and input tensor to the same device (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Perform inference with the U-Net model
with torch.no_grad():
    output = model(input_tensor)

# Post-process the output
output = output.cpu().numpy()  # Move to CPU and convert to NumPy array
segmentation_mask = np.argmax(output, axis=1)  # If your model outputs class probabilities, use argmax to get the class with the highest probability

# Display the original image and processed segmentation mask side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")

# Processed segmentation mask
axes[1].imshow(segmentation_mask[0], cmap='jet')  # Adjust colormap as needed
axes[1].set_title("Segmentation Mask")

plt.show()
