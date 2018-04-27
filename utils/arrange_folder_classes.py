import os
import shutil
import numpy as np

from PIL import Image


def arrange_images_for_image_folder(in_path, out_path):
    for alphabet in os.listdir(in_path):
        base_name = alphabet + '_'
        alphabet_path = os.path.join(in_path, alphabet)
        for character in os.listdir(alphabet_path):
            out_file_name = base_name + '_' + character
            character_path = os.path.join(alphabet_path, character)
            out_character_path = os.path.join(out_path, out_file_name)
            try:
                os.makedirs(out_character_path)
            except OSError:
                pass
            for im in os.listdir(character_path):
                os.rename(os.path.join(character_path, im), os.path.join(out_character_path, im))


def resize_images_and_save(in_path, size):

    out_path = in_path + '_' + str(size[0]) + 'x' + str(size[1])
    try:
        os.makedirs(out_path)
    except OSError:
        pass

    for image_class in os.listdir(in_path):

        image_class_path = os.path.join(in_path, image_class)
        out_class_path = os.path.join(out_path, image_class)
        try:
            os.makedirs(out_class_path)
        except OSError:
            pass

        for img in os.listdir(image_class_path):
            img_path = os.path.join(image_class_path, img)
            out_img_path = os.path.join(out_class_path, img)
            im = Image.open(img_path)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(out_img_path)


def copy_random_dirs(in_path, out_path, n):
    all_dirs = os.listdir(in_path)
    to_copy = np.random.choice(all_dirs, n, replace=False)
    for d in to_copy:
        out_dir = os.path.join(out_path, d)
        os.makedirs(out_dir)
        curr_dir = os.path.join(in_path, d)
        for f in os.listdir(curr_dir):
            shutil.copyfile(os.path.join(curr_dir, f), os.path.join(out_dir, f))
            print(1)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


os.chdir(os.path.join(os.getcwd(), os.pardir))
copy_random_dirs('omniglot/images_background', 'omniglot/train_data1', 80)
copy_random_dirs('omniglot/images_background', 'omniglot/train_data2', 80*3)
copy_random_dirs('omniglot/images_background', 'omniglot/train_data3', 80*5)
# resize_images_and_save('images_background_out', (64, 64))
