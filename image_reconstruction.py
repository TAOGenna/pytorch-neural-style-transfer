"""
FEATURE MAP VISUALIZATOR
"""
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    print(type(image))
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    return img_transform(image).unsqueeze(0)

# Function to convert a tensor back to an image
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    # Un-normalize the image
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    # Clip the pixel values to be between 0 and 1
    image = image.clip(0, 1)
    return image

def get_features(img, model):
    pass
# Fetch image and convert it to a torch tensor
content_img_path = os.path.join(os.path.dirname(__file__), 'data', 'content', 'bariloche.jpg')
img = load_image(content_img_path).to(DEVICE)
# --------------------------------------------------------------------------------------


# -------------------- Define vgg19 net and useful layers ---------------------------------
# tips: use `nametuple` 
# use features: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
# {'features.3':'conv1','features.10':'conv2','features.23':'conv3','features.36':'conv4','features.49':'conv5'}
vgg_net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True);
vgg_net.eval().to(DEVICE)
# as stated in the paper, we extract the layers using a CNN
model = create_feature_extractor(vgg_net,{'features.49':'conv5'})

# Optimization part
with torch.no_grad():
    fixed_out = model(img)['conv5'].detach() # for now we select conv5, but it can be any of the other convs
# initialize noise 
white_noise = torch.rand(size=img.shape, requires_grad=True, device=DEVICE)

# optimization setup
loss_function = nn.MSELoss()
epochs = 2000
optimizer = torch.optim.Adam([white_noise]),lr=0.003)

for i in (t := trange(epochs)):
    optimizer.zero_grad()
    # At each epoch, we send through our newly `white_noise` image
    end_image = model(white_noise)['conv5']
    loss = loss_function(end_image, fixed_out)
    loss.backward()
    optimizer.step()
    t.set_description(f'loss function = {loss:.6f}')
    
    if i % 50 == 0:
        reconstructed_image = im_convert(white_noise)
        image = Image.fromarray((reconstructed_image * 255).astype('uint8'))
        image.save('output_image.jpg')
     
