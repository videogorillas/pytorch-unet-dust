from random import Random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
from torchsummary import summary

from PIL import Image
from torch.nn import BCELoss, CrossEntropyLoss

from unet import UNet


class FilmDustDataset(data.Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        listdir = os.listdir(rootdir)
        alphapaths = list(filter(lambda n: n.endswith("_alpha.png"), listdir))
        paths = list(filter(lambda n: n.endswith(".png") and not n.endswith("_alpha.png"), listdir))
        da = {i.replace("_alpha.png", ""): i for i in alphapaths}
        dp = {i.replace(".png", ""): i for i in paths}

        self.okpaths = list(map(lambda k: dp[k], (filter(lambda k: k in dp, da.keys()))))
        self.rnd = Random(42434445)
        self.totensor = transforms.ToTensor()
        self.dilation_kernel = np.ones((3, 3), np.uint8)

    def __getitem__(self, index):
        idx = index
        path = self.okpaths[idx]
        apath = path.replace(".png", "_alpha.png")
        # print(path, ":", apath)
        cimg = Image.open(os.path.join(self.rootdir, path))
        cmask = Image.open(os.path.join(self.rootdir, apath))

        timg = self.totensor(cimg)
        dust = self.totensor(cmask)
        cmasknp = np.array(cmask, dtype=np.float32)
        dilated = cv2.dilate(cmasknp, self.dilation_kernel, iterations=1) / 255.0
        cmasksum = (cmasknp / 255.0).sum()
        dilatedsum = dilated.sum()
        while dilatedsum / cmasksum < 2:
            dilated = cv2.dilate(dilated, self.dilation_kernel, iterations=1)
            dilatedsum = dilated.sum()

        dt = self.totensor(dilated)
        return {'img': timg, 'mask': dust, 'dilated': dt, 'dilated_pixcnt': dt.sum()}

    def __len__(self):
        return len(self.okpaths)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        tensor0 = None
        if (len(image_tensor.shape) == 3):
            tensor0 = image_tensor
        else:
            tensor0 = image_tensor[0]
        image_numpy = tensor0.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        #        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

_W = 256
_H = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = UNet(in_channels=3, wf=3, depth=3, n_classes=1, padding=True, up_mode='upconv', batch_norm=True).to(device)
print(model)
summary(model, input_size=(3, _W, _H))

optim = torch.optim.Adam(model.parameters())

filmdust = FilmDustDataset("/home/zhukov/clients/uk/dustdataset/ok/256.e4d4")
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
        dsumn = dsum[n].item() / 2
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

for e in range(20):
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
