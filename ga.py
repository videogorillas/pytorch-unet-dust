import torch
from torchsummary import summary

from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=3, wf=4, n_classes=1, padding=True, up_mode='upconv').to(device)
print(model)
summary(model, input_size=(3, 64, 64))

# todo
# 64x64