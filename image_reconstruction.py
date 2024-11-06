"""
FEATURE MAP VISUALIZATOR
"""
import torch
import torch.nn as nn
import os
from torchvision import transforms
import cv2 as cv
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# ----------------- Fetch image and convert it to a torch tensor ---------------------

content_img_path = os.path.join(os.path.dirname(__file__),'data','content','bariloche.jpg')
img = cv.imread(content_img_path)[:,:,::-1]

# uncomment to visualize image
# cv.imshow('image',img[:,:,::-1])
# cv.waitKey(0)

img = img.astype(np.float32)
TARGET_SIZE = 256
img = np.resize(img,(TARGET_SIZE,TARGET_SIZE,3))
img /= 255

transformations = transforms.Compose([
    transforms.ToTensor()
])

img = transformations(img).unsqueeze(0)
# --------------------------------------------------------------------------------------


# -------------------- Define vgg19 net and useful layers ---------------------------------
# tips: use `nametuple` 
# use features: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
# {'features.3':'conv1','features.10':'conv2','features.23':'conv3','features.36':'conv4','features.49':'conv5'}
vgg_net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True);
vgg_net.eval();
# as stated in the paper, we extract the layers using a CNN
model = create_feature_extractor(vgg_net,{'features.49':'conv5'})

# -------------------------------------------------------------------------------------

# ------------------- sanity check: image through feature maps ----------------------------
out = model(img)

def sanity_check():
    for i in range(1,6):
        display_image = out[f'conv{i}'][0,0,:,:].detach().cpu().numpy()
        plt.imshow(display_image)
        plt.axis('off')
        plt.show()

#sanity_check()


# define white noise
white_noise = torch.rand(size=img.shape, requires_grad=True)

# Optimization part
epochs = 3000
optimizer = torch.optim.Adam((white_noise,),lr=1e1)
fixed_out = out['conv5'] # for now we select conv5, but it can be any of the other convs
loss_function = torch.nn.MSELoss()
for i in (t := trange(epochs)):
    # at each epoch we send through our newly `white_noise` image
    end_image = model(white_noise)['conv5']
    loss = loss_function(end_image, fixed_out)
    loss.backward(retain_graph = True)
    optimizer.step()
    optimizer.zero_grad()
    t.set_description(f'loss function = {loss:.2f}')


