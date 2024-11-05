import torch


# inputs are style_img and content_img
# TRANSFORM IMAGES to some target dimensions
def load_image():
    pass

content_path = ''
content_img = load_image()

# define model | vgg19 
def vgg19():
    pass

# I pass both style_img and content_img through the vgg19 net to obtain the static data
# `feature_map_....` is a list containing different instance of feature maps at different layers
feature_map_content_img = vgg19(content_img)

#
# Optimization part
#
epochs = 3000

# use adam optimizer | vgg has `required_grad=False`
# we only want the derivative w.r.t the pixels from the white noise image so that is the only learnable parameter we specify in the optimizer | this is mentioned in fig2 of original paper
optimizer = None

def loss_function():
    pass

for i in range(epochs):
    pass


