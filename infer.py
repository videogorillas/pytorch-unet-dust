import argparse
import os

import torch
import torchvision

from unet import UNet
from util import save_image, tensor2im, imread16

parser = argparse.ArgumentParser(description='Detect film dust')
parser.add_argument('--weights_path', type=str, default='weights/dust8_19.pth')
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('paths', metavar='paths', type=str, nargs='+', help='paths to images')

cfg = parser.parse_args()
print(cfg)
device = torch.device(cfg.device)
print("using", device)
model = UNet(in_channels=3, wf=4, depth=4, n_classes=1, padding=True, up_mode='upconv', batch_norm=True).to(device)
model.load_state_dict(torch.load(cfg.weights_path), strict=True)
model.to(device)
model.eval()

totensor = torchvision.transforms.ToTensor()

with torch.no_grad():
    for path in cfg.paths:
        print(path)
        basename = os.path.basename(path)
        img = imread16(path)
        t = totensor(img).to(device).unsqueeze(0)
        predicted = model(t).detach()
        save_image(tensor2im(predicted[0]), cfg.output_dir + "/" + basename)
