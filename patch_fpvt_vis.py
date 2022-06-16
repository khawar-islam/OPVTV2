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
from vit_pytorch.ours import PyramidVisionTransformerV2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
# create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

model = PyramidVisionTransformerV2(
    loss_type='ArcMarginProduct',
    GPU_ID=0,
    img_size=112,
    depths=[3, 4, 18, 3],
    patch_size=8,
    num_classes=526,
    in_chans=3
)

path = 'results/ViT-P12S8_ms1m_cosface_s1/Backbone_PVTV2_Epoch_1_Batch_920_Time_2022-06-16-15-51_checkpoint.pth'
model.load_state_dict(torch.load(path))
# model.eval()


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
#img = img.resize((112, 112), Image.ANTIALIAS)
img_tensor = transforms(img).unsqueeze(0)
print(img_tensor.shape)

# end-to-end inference
output = model(img_tensor)

print("Inference Result:")
print("Face")
plt.imshow(img)

# 1. Split Image into Patches The input image is split into N patches (N = 14 x 14 for ViT-Base) and converted to
# D=768=16x16x3 embedding vectors by learnable 2D convolution: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
patches = model.patch_embed1(img_tensor)  # patch embedding convolution

print("Image tensor: ", img_tensor.shape)
# Image tensor:  torch.Size([1, 3, 112, 112])

# print("Patch embeddings: ", patches.shape)
# Patch embeddings:  torch.Size([1, 196, 768])

# This is NOT a part of the pipeline.
# Actually the image is divided into patch embeddings by Conv2d
# with stride=(16, 16) shown above.
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of patch_embed1", fontsize=24)
fig.add_axes()
img = np.asarray(img)

# # Patch stage 1
# for i in range(0, 784):  # 28 28 (number of patches in width and height) 112/4=28
#     x = i % 28
#     y = i // 28
#     patch = img[y * 4:(y + 1) * 4, x * 4:(x + 1) * 4]
#     ax = fig.add_subplot(28, 28, i + 1)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.imshow(patch)
#
# plt.savefig('patch_embed1.png', figsize=(112, 112))

# Patch stage 1
for i in range(0, 784):  # 28 28 (number of patches in width and height) 112/4=28
    x = i % 28
    y = i // 28
    patch = img[y * 4:(y + 1) * 4, x * 4:(x + 1) * 4]
    ax = fig.add_subplot(28, 28, i + 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)

plt.savefig('patch_embed1.png', figsize=(112, 112))