import torch
from torchsummary import summary
from dataset import FilmDustDataset, FilmDustSeqDataset

from unet import UNet
from util import save_image, tensor2im

_W = 256
_H = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# model = UNet(in_channels=9, wf=4, depth=3, n_classes=1, padding=True, up_mode='upconv', batch_norm=True).to(device)
model = UNet(in_channels=3, wf=3, depth=3, n_classes=1, padding=True, up_mode='upconv', batch_norm=True).to(device)
print(model)
summary(model, input_size=(3, _W, _H))
# model.load_state_dict(torch.load('weights/w3d3/dust_19.pth'), strict=False)
model.load_state_dict(torch.load('weights/dustseq_19.pth'), strict=False)
model.to(device)
model.eval()

# dir = "/home/zhukov/clients/uk/dustdataset/ok/768x256.e4d4"
# dataset = FilmDustSeqDataset(dir)
dir = "/home/zhukov/clients/uk/dustdataset/gimar/256.e4d4"
# dir = "/home/zhukov/clients/uk/dustdataset/ok/256.e4d4"
dataset = FilmDustDataset(dir)
for d in dataset:
    x = d['img'].to(device).unsqueeze(0)
    expected = d['mask'].unsqueeze(0)
    fname = d['path']
    print(fname)
    predicted = None
    with torch.no_grad():
        predicted = model(x)

    save_image(tensor2im(predicted), "%s/%s_pred.png" % (dir, fname))

    # save_image(tensor2im(x), "tmp/img.png")
    # save_image(tensor2im(predicted), "tmp/pred.png")
    # save_image(tensor2im(expected), "tmp/expected.png")
