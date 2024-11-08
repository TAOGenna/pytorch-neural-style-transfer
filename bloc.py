import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size
imsize = 512 if torch.cuda.is_available() else 256  # Use small size if no GPU

# Image loading and preprocessing function with normalization
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load the texture image
texture_img = image_loader("/home/rotakagui/projects/pytorch-neural-style-transfer/data/style/starry_night.jpg")

# Function to convert a tensor back to an image
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    # Un-normalize the image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # Clip the pixel values to be between 0 and 1
    image = np.clip(image, 0, 1)
    return image

# Define the Gram Matrix function
def gram_matrix(input):
    _, feature_maps, h, w = input.size()
    features = input.view(feature_maps, h * w)
    G = torch.mm(features, features.t())
    # Normalize the Gram matrix
    return G.div(feature_maps * h * w)

# Load the pre-trained VGG19 model
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Freeze all VGG parameters since we're optimizing the input image
for param in cnn.parameters():
    param.requires_grad = False

# Define the layers to be used for texture representation
texture_layers = [0, 5, 10, 19, 28]  # Indices of the layers

# Extract features of the texture image at the desired layers
def get_features(image, model, layers):
    features = []
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features

# Get the target features
target_features = get_features(texture_img, cnn, texture_layers)
target_grams = [gram_matrix(f) for f in target_features]

# Initialize the input image (white noise)
input_img = torch.randn(texture_img.data.size(), device=device)
input_img = input_img * 0.001  # Small initialization
input_img.requires_grad_(True)

# Define optimizer
optimizer = optim.Adam([input_img], lr=0.1)

# Run the texture synthesis
num_steps = 7000
print('Optimizing...')

for step in range(1, num_steps + 1):
    optimizer.zero_grad()

    # Forward pass
    features = get_features(input_img, cnn, texture_layers)
    loss = 0
    for f, t in zip(features, target_grams):
        G = gram_matrix(f)
        loss += nn.functional.mse_loss(G, t)

    # Backward pass
    loss.backward()

    # Update image
    optimizer.step()

    # Clamp the input image to valid range after optimizer step
    with torch.no_grad():
        # Since we normalized the input, we need to clamp using the normalized values
        for c in range(3):
            min_val = (0 - loader.transforms[2].mean[c]) / loader.transforms[2].std[c]
            max_val = (1 - loader.transforms[2].mean[c]) / loader.transforms[2].std[c]
            input_img.data[0, c].clamp_(min_val, max_val)

    if step % 50 == 0:
        print("Step {}: Texture Loss : {:6f}".format(step, loss.item()))
        output_image = im_convert(input_img)
        output_pil = Image.fromarray((output_image * 255).astype('uint8'))
        output_pil.save('starry_night_synthesized.jpg')

# Save the final output image
output_image = im_convert(input_img)
output_pil = Image.fromarray((output_image * 255).astype('uint8'))
output_pil.save('starry_night_synthesized.jpg')
