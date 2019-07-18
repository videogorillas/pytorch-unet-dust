import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.nn import BCELoss

from dataset import FilmDustDataset
from unet import UNet
from util import save_image, tensor2im

_W = 256
_H = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = UNet(in_channels=3, wf=3, depth=3, n_classes=1, padding=True, up_mode='upconv', batch_norm=True).to(device)

model.load_state_dict(torch.load('weights/w3d3/dust_19.pth'), strict=False)
model.to(device)

optim = torch.optim.Adam(model.parameters())

filmdust = FilmDustDataset("/home/zhukov/clients/uk/dustdataset/gimar/256.e4d4/finetune")
print(len(filmdust))
dataloader = torch.utils.data.DataLoader(
    filmdust,
    batch_size=42,
    shuffle=True,
    num_workers=8)

# dilate mask
# tensorboard or visdom
# lossf = CrossEntropyLoss()
# lossf = BCELoss(reduction='none')
lossf = BCELoss()

dilation_kernel = np.ones((3, 3), np.uint8)


def falsepositives_mask(prediction, dilated):
    batch_size = dilated.shape[0]
    falsepositives = (prediction - dilated)
    fpmask = (falsepositives > 0)
    falsepositives = falsepositives * fpmask.float()
    fpmax = (falsepositives.reshape(batch_size, _W * _H).max(1).values * 255.0).int()
    dsum = dilated.reshape(batch_size, _W * _H).sum(1).int()
    for n, fp in enumerate(falsepositives):
        fpmaxn = fpmax[n].item()
        dsumn = dsum[n].item()
        dilatedsum = 0
        for m in range(fpmaxn, -1, -1):
            fpmaskn = fp > (m / 255.0)
            fpmasknd = fpmaskn.cpu().squeeze().detach().numpy()
            fpmasknd = cv2.dilate(fpmasknd, dilation_kernel, iterations=1)
            dilatedsum = fpmasknd.sum()
            # print(dilatedsum, dsumn)
            if dilatedsum >= dsumn:
                break

        fpmaskn = torch.tensor(fpmasknd, device=device).float()
        falsepositives[n] = fp * fpmaskn + dilated[n]
    mask = (dilated + (falsepositives > 0).float()).clamp(0.0, 1.0)
    return (falsepositives, mask)


for e in range(10):
    for i, batch in enumerate(dataloader):
        img = batch['img'].to(device)
        expected = batch['mask'].to(device)
        dilated = batch['dilated'].to(device)
        prediction = model(img)

        falsepositives, mask = falsepositives_mask(prediction, dilated)

        ga = prediction * mask.detach()
        loss = lossf(ga, expected)
        # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/4
        # pixelcnt = batch['dilated_pixcnt'].to(device)
        # pixelcnt = pixelcnt.reshape((loss.shape[0], 1, 1, 1))
        # w = (pixelcnt / pixelcnt.sum()).detach()
        # theloss = (loss * w).mean()

        # maskedloss = loss * dilated
        # theloss = torch.sum(maskedloss) / torch.sum(dilated)

        print("epoch", e, "iter", i, "loss", loss.item(), "min", int(prediction.min().item() * 255), "max",
              int(prediction.max().item() * 255))
        # print("epoch", e, "iter", i, "loss", theloss.item(), "min", int(prediction.min().item() * 255), "max",
        #       int(prediction.max().item() * 255))
        if i % 10 == 0:
            img0 = img[0]
            expected0 = expected[0]
            edir = 'tmp/' + str(e)
            if not os.path.isdir(edir):
                os.mkdir(edir)

            save_image(tensor2im(img0), "%s/%03d_0img.png" % (edir, i))
            save_image(tensor2im(torch.stack([prediction[0]], 0)), "%s/%03d_1pred.png" % (edir, i))
            save_image(tensor2im(torch.stack([expected0[0]], 0)), "%s/%03d_2expected.png" % (edir, i))
            save_image(tensor2im(torch.stack([dilated[0]], 0)), "%s/%03d_2dilate.png" % (edir, i))
            save_image(tensor2im(torch.stack([mask[0]], 0)), "%s/%03d_3mask.png" % (edir, i))
            save_image(tensor2im(torch.stack([falsepositives[0]], 0)), "%s/%03d_4falsepositives.png" % (edir, i))
            # save_image(tensor2im(torch.stack([prediction[0][0]], 0)), edir + "/" + str(i) + "_pred0.png")
            # save_image(tensor2im(torch.stack([(prediction[0][1])], 0)), edir + "/" + str(i) + "_pred1.png")
            print(i)
        optim.zero_grad()
        # theloss.backward()
        loss.backward()
        optim.step()
    torch.save(model.state_dict(), 'weights/finetune_dust_' + str(e) + '.pth')
