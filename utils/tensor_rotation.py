import numpy as np
import torch


def rotate_tensor(img, rotation_num=0, flip=False):

    rotated_image = np.rot90(img, rotation_num, axes=(1, 2))
    if flip:
        rotated_image = np.flipud(rotated_image)

    rotated_image = torch.from_numpy(rotated_image.copy())
    return rotated_image
