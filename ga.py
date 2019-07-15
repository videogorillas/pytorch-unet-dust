import torch
from torchsummary import summary

from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=9, wf=4, depth=3, n_classes=1, padding=True, up_mode='upconv', batch_norm=True).to(device)
print(model)
summary(model, input_size=(9, 256, 256))

# todo
# 64x64