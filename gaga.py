import cv2
import numpy as np
from PIL import Image

img = cv2.imread('/home/zhukov/tmp/ok/R3B_108419_0172_alpha.png', cv2.IMREAD_GRAYSCALE)
y = 0
x = 0
w = 256
h = 256
crop = img #[y:y + h, x:x + w]
print(crop)
kernel = np.ones((5, 5), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(crop, kernel, iterations=1)
erode = cv2.erode(crop, kernel2, iterations=1)
opening = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel3)
opening2 = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel2)

cv2.imwrite('tmp/dilate.png', dilation)
cv2.imwrite('tmp/crop.png', crop)
cv2.imwrite('tmp/erode2.png', erode)
cv2.imwrite('tmp/open.png', opening)
cv2.imwrite('tmp/open2.png', opening2)

# img = Image.open('/home/zhukov/tmp/ok/R2A_108248_0156_alpha.png')
# crop = img.crop((0, 0, 256, 256))
a = np.array(dilation, dtype=float) / 255.0
print(a.shape)
print(a.max())
print(a.min())
# print(img)
# print(crop)
# img.crop(0, 0, 16, 16)
