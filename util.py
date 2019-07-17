import cv2
import torch
import numpy as np
from PIL import Image


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
    cv2.imwrite(image_path, image_numpy)
    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path)


# reads image in 16bit per channel mode if possible, returns ndarray (h,w,c) normalized to [0.0..1.0]
def imread16(path: str) -> np.ndarray:
    flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    src = cv2.imread(path, flags)
    assert src.dtype == np.uint16 or src.dtype == np.uint8, "unhandled data type " + str(src.dtype)

    max_val = 65535. if src.dtype == np.uint16 else 255.
    return src.astype("float32") / max_val
