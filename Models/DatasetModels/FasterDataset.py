import PIL.ImageOps
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.tensor_rotation import rotate_tensor


class SiameseNetworkDataset(Dataset):
    """
    Faster structure but with some (minor) problems

    M - number of classes
    K - number of samples in each class (doesnt always exist)
    N - total number of images N = K * M
    Np - number of unique positive pairs Np = M * K * (K - 1) / 2 (in case such K exist)
    Nn - number of unique negative pairs Nn = (N ^ 2 - N) / 2 - Np

    Pros:
        - Exactly 1/1 samples in each batch
        - In each epoch every positive pair is given exactly once
        - Fast pre-process (O(Np))
        - Shuffling is faster (O(Np))
        - less space (O(Np))
    Cons:
        - negative pairs may duplicate
        - P[draws are NOT unique in one epoch] = 1 -  Nn! / ((Nn - Np + 1)! * Nn ^ (Np - 1))
        - E[number of unique draws in one epoch] = Nn - E[number of not picked elements] =
            = Nn - Nn * P[not picking specific element in epoch] = Nn - Nn * (1 - 1/Nn) ^ Np

    """

    def __init__(self, images_folder, transform=None, should_invert=True,
                 duplicate_image=False, load_images=False, include_rotations=False):
        self.images_folder = images_folder
        self.transform = transform
        self.should_invert = should_invert
        self.num_images = len(images_folder.imgs)
        self.num_classes = len(images_folder.classes)
        self.include_rotations = include_rotations

        images = images_folder.imgs
        len_images = len(images)

        positive_pairs = []
        positive_pairs_range = []
        start_index = 0
        for i in range(len_images):
            # if we change class change the start index
            if images[i][1] != images[start_index][1]:
                start_index = i

            if duplicate_image:
                r = range(i, len_images)
            else:
                r = range(i + 1, len_images)

            for j in r:

                # if image of the same class insert to positive pairs
                if images[i][1] == images[j][1]:
                    positive_pairs.append((i, j, 0))
                    if include_rotations:
                        positive_pairs.append((i, j, 1))
                        positive_pairs.append((i, j, 2))
                        positive_pairs.append((i, j, 3))
                        positive_pairs.append((i, j, 4))
                        positive_pairs.append((i, j, 5))
                        positive_pairs.append((i, j, 6))
                        positive_pairs.append((i, j, 7))

                # else insert to negative range and end cycle
                else:
                    positive_pairs_range.append((start_index, j))
                    break

            # if we are in last class add last class range
            if images[-1][1] == images[start_index][1]:
                positive_pairs_range.append((start_index, len_images))

        self.num_positives = len(positive_pairs)
        self.positive_pairs = np.array(positive_pairs)
        self.positive_pairs_range = np.array(positive_pairs_range)

        # loading images if needed
        self.loaded = False
        self.loaded_images = []
        if load_images:
            self.preload_images()
            self.loaded = True

    def __getitem__(self, index, rotate=False):

        # when starting epoch shuffle
        if index == 0:
            self.shuffle()

        # choosing indices
        indices = self.choose_indices(index)

        # handling the 2 chosen images
        img0 = self.get_image(indices[0], indices[2])
        img1 = self.get_image(indices[1], indices[2])

        return img0, img1, torch.from_numpy(np.array([int(index % 2)], dtype=np.float32))

    def choose_indices(self, index):
        # if index is odd - give random negative example
        if index % 2:
            rotation = 0
            if self.include_rotations:
                rotation = np.random.randint(0, 8)
            first_image = np.random.randint(0, self.num_images)

            class_start = self.positive_pairs_range[first_image][0]
            class_end = self.positive_pairs_range[first_image][1]

            # special case: we picked the first class
            if class_start == 0:
                second_image = np.random.randint(class_end, self.num_images)

            # special case: we picked the last class
            elif class_end == self.num_images:
                second_image = np.random.randint(0, class_start)
                first_image, second_image = second_image, first_image

            # else need to give probabilities to before and after choices
            else:
                num_class_images = class_end - class_start
                num_overall_choices = self.num_images - num_class_images
                second_image = np.random.randint(0, num_overall_choices)

                # fix second image if needed
                if second_image >= class_start:
                    second_image += num_class_images

                # else first image is larger for sure
                else:
                    first_image, second_image = second_image, first_image

            indices = (first_image, second_image, rotation)

        # else index is even - give positive example
        else:
            indices = self.positive_pairs[int(index / 2)]

        return indices

    def shuffle(self):
        np.random.shuffle(self.positive_pairs)

    def preload_images(self):
        for i in range(self.num_images):
            self.loaded_images.append(self.get_image(i))

    def get_image(self, index, rotation=0):
        # if we already loaded all images
        if self.loaded:
            img = self.loaded_images[index]
            if rotation > 0:
                flip = False
                if rotation > 4:
                    flip = True
                    rotation -= 4
                img = rotate_tensor(img, rotation, flip=flip)
            return img

        # else load and preprocess
        img = Image.open(self.images_folder.imgs[index][0])
        img = img.convert("L")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return 1 - img

    def not_drawing_unique(self):
        num_negatives = self.get_number_negatives()
        num_positives = self.num_positives
        start = num_negatives - num_positives + 1
        finish = start + num_positives
        p = 0
        for i in range(start, finish):
            p += np.log(i/num_negatives)
        return np.exp(p)

    def expected_unique_draws(self):

        num_negatives = self.get_number_negatives()
        num_positives = self.num_positives
        prob_not_picking_element_in_epoch = (1 - 1/num_negatives) ** num_positives
        expected_num_unpicked_elements_in_epoch = num_negatives * prob_not_picking_element_in_epoch
        return num_negatives - expected_num_unpicked_elements_in_epoch

    def get_number_negatives(self):
        return int((self.num_images ** 2 - self.num_images) / 2 - self.num_positives)

    def __len__(self):
        return 2 * self.num_positives
