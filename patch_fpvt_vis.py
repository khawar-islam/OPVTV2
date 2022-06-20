import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from timm import create_model
from PIL import Image
from vit_pytorch.ours import Ours_FPVT

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device = ", device)

'''
Original implementation
image_Size=224,224
1) Split Image into Patches 
    The input image is split into 14 x 14 vectors with dimension of 768 by Conv2d (k=16x16) with stride=(16, 16). 
2) Add Position Embeddings 
    Learnable position embedding vectors are added to the patch embedding vectors and fed to the transformer encoder. 
3) Transformer Encoder 
    The embedding vectors are encoded by the transformer encoder. The dimension of input and output vectors are the same.
4) MLP (Classification) 
    Head The 0th output from the encoder is fed to the MLP head for classification to output the final  classification results. 
'''

path = 'results/ViT-P12S8_ms1m_cosface_s1/Backbone_PVTV2_Epoch_1_Batch_920_Time_2022-06-16-15-51_checkpoint.pth'

model = Ours_FPVT(
    loss_type='ArcMarginProduct',
    GPU_ID=0,
    img_size=112,
    depths=[3, 4, 18, 3],
    patch_size=8,
    num_classes=526,
    in_chans=3
)
model.load_state_dict(torch.load(path))

IMG_SIZE = (112, 112)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
]

transforms = T.Compose(transforms)

# Demo Image
img = PIL.Image.open('Adam_Brody_233.png')
img = img.resize((112, 112), Image.ANTIALIAS)
img_tensor = transforms(img).unsqueeze(0)

'''
# end-to-end inference
output = model.patch_embed1(img_tensor)

# Patch embed 1
patches = model.patch_embed1(img_tensor)  # patch embedding convolution
print("Image tensor: ", patches)  # 28x28=784
# Image tensor:  torch.Size([1, 3, 112, 112])

print("Patch embeddings: ", model.patch_embed1)
# this value coming from forward pass of Ours_FPVT torch.Size([1, 784, 64])

fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of patch_embed1", fontsize=24)
fig.add_axes()
img = np.asarray(img)

# Patch embed 1
for i in range(0, 784):
    x = i % 28
    y = i // 28
    patch = img[y * 4:(y + 1) * 4, x * 4:(x + 1) * 4]
    ax = fig.add_subplot(28, 28, i + 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)
plt.savefig('patch_embed1.png', figsize=(112, 112))


# patch embed 2
model.patch_embed2.img_size = 112 // 4
model.patch_embed2.patch_size = 3
model.patch_embed2.stride = 2
model.patch_embed2.in_chans = 64
model.patch_embed2.embed_dim = 128

patches = model.patch_embed2
print(patches.H)

# torch.Size([1, 196, 128])
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of patch_embed2", fontsize=24)
fig.add_axes()
img = np.asarray(img)

# model.patch_embed2.proj----> Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# Patch embed 2
for i in range(0, 196):  # 28 28 (number of patches in width and height) 112/4=28
    x = i % 14
    y = i // 14
    patch = img[y * 2:(y + 1) * 2, x * 2:(x + 1) * 2]
    ax = fig.add_subplot(14, 14, i + 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)
plt.savefig('patch_embed2.png', figsize=(112, 112))


output = model.patch_embed1(img_tensor)
# patch embed 3
model.patch_embed3.img_size = 112 // 8
model.patch_embed3.patch_size = 3
model.patch_embed3.stride = 2
model.patch_embed3.in_chans = 128
model.patch_embed3.embed_dim = 256

patches = model.patch_embed3
print(patches.H)

# torch.Size([1, 196, 128])
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of patch_embed3", fontsize=24)
fig.add_axes()
img = np.asarray(img)

# model.patch_embed3.proj----> Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# Patch embed 2
# forwaRD PASS  torch.Size([1, 49, 256])

for i in range(0, 49):  # 7 7 (number of patches in width and height) 112/4=28
    x = i % 7
    y = i // 7
    patch = img[y * 2:(y + 1) * 2, x * 2:(x + 1) * 2]
    ax = fig.add_subplot(7, 7, i + 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)
plt.savefig('patch_embed3.png', figsize=(112, 112))
'''

output = model.patch_embed1(img_tensor)
# patch embed 3
model.patch_embed4.img_size = 112 // 16
model.patch_embed4.patch_size = 3
model.patch_embed4.stride = 2
model.patch_embed4.in_chans = 256
model.patch_embed4.embed_dim = 512

patches = model.patch_embed3
print(patches.H)

# torch.Size([1, 196, 128])
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of patch_embed3", fontsize=24)
fig.add_axes()
img = np.asarray(img)

# model.patch_embed4.proj----> Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# Patch embed 2
# stage 4: torch.Size([1, 16, 512])

for i in range(0, 16):  # 7 7 (number of patches in width and height) 112/4=28
    x = i % 4
    y = i // 4
    patch = img[y * 2:(y + 1) * 2, x * 2:(x + 1) * 2]
    ax = fig.add_subplot(4, 4, i + 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)
plt.savefig('patch_embed4.png', figsize=(112, 112))