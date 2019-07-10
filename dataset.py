from random import Random

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os

from PIL import Image


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
        assert cmasksum > 0, "no pixels in mask " + apath
        dilatedsum = dilated.sum()
        while dilatedsum / cmasksum < 2:
            dilated = cv2.dilate(dilated, self.dilation_kernel, iterations=1)
            dilatedsum = dilated.sum()
            if dilatedsum == cmasksum:  # dilation impossible
                break

        dt = self.totensor(dilated)
        return {'img': timg, 'mask': dust, 'dilated': dt, 'dilated_pixcnt': dt.sum(), 'path': path}

    def __len__(self):
        return len(self.okpaths)
