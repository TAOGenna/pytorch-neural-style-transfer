import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
from tqdm import tqdm, trange

# Function to load and preprocess the image
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    
    # Resize the image
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
        
    if shape is not None:
        size = shape
    
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # Normalize using ImageNet's mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])])
    
    # Apply the transform to the image and add a batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# Function to extract features from the model
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation layer
            '28': 'conv5_1'
        }
    features = {}
    x = image
    # Iterate through the model's layers
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Function to convert a tensor back to an image
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    # Un-normalize the image
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    # Clip the pixel values to be between 0 and 1
    image = image.clip(0, 1)
    return image

# Load the pre-trained VGG19 model
vgg = models.vgg19(pretrained=True).features

# Freeze all model parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Load the content image
content_image_path = os.path.join(os.path.dirname(__file__), 'data', 'content', 'bariloche.jpg')  # Replace with your image path
content = load_image(content_image_path).to(device)

# Get the content features
content_features = get_features(content, vgg)

# Create a white noise image
target = torch.randn_like(content).requires_grad_(True).to(device)

# Define optimizer and hyperparameters
optimizer = optim.Adam([target], lr=0.003)
content_weight = 1  # Weight for content loss
num_steps = 2000  # Number of iterations

# Optimization loop
for step in (t := trange(1, num_steps + 1)):
    # Extract features from the target image
    target_features = get_features(target, vgg)
    # Calculate content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    # Total loss
    loss = content_weight * content_loss
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 steps
    if step % 100 == 0:
        reconstructed_image = im_convert(target)
        reconstructed_image = Image.fromarray((reconstructed_image*255).astype('uint8'))
        reconstructed_image.save('reco_image.jpg')
        t.set_description(f'loss function = {loss.item():.6f}')
        #print(f'Step {step}, Loss: {loss.item()}')

# Display the reconstructed image
plt.figure(figsize=(10, 10))
plt.imshow(im_convert(target))


from PIL import Image
# Convert the tensor to a NumPy array and unnormalize it
reconstructed_image = im_convert(target)

# Convert the NumPy array to a PIL Image
image = Image.fromarray((reconstructed_image * 255).astype('uint8'))

# Save the image to a file
image.save('reconstructed_image.jpg')

plt.axis('off')