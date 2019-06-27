import cv2
import numpy as np
from PIL import Image

img = cv2.imread('/home/zhukov/tmp/ok/R2A_108248_0156_alpha.png', cv2.IMREAD_GRAYSCALE)
y = 0
x = 0
w = 256
h = 256
crop = img[y:y + h, x:x + w]
print(crop)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(crop, kernel, iterations=1)
cv2.imwrite('tmp/dilate.png', dilation)
cv2.imwrite('tmp/crop.png', crop)

# img = Image.open('/home/zhukov/tmp/ok/R2A_108248_0156_alpha.png')
# crop = img.crop((0, 0, 256, 256))
a = np.array(dilation, dtype=float) / 255.0
print(a.shape)
print(a.max())
print(a.min())
# print(img)
# print(crop)
# img.crop(0, 0, 16, 16)
