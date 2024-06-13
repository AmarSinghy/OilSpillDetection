import os
import random
import numpy as np
from PIL import Image


class DataPreparation:
    """Class for preparing data for segmentation tasks."""

    def __init__(self, img_size=(256, 256)):
        """
        Initialize the DataPreparation class.

        Args:
            img_size (tuple): Image size (height, width).
        """
        self.img_size = img_size

        self.color_to_index, self.index_to_value = self.generate_target_ids()
        self.value_to_index = {tuple(v): k for k, v in self.index_to_value.items()}

    def generate_target_ids(self, colors_to_rgb_txt='D:\Color to RGB Mapping.txt'):
        """
        Generate a mapping of colors to class indices and class indices to RGB values.

        Args:
            colors_to_rgb_txt (str): Path to the file containing the color to RGB mapping.

        Returns:
            tuple: Color to index mapping, index to value mapping.
        """
        color_to_index_map = {}
        index_to_value_map = {}

        with open(colors_to_rgb_txt, 'r') as f:
            lines = f.readlines()
            for id_, line in enumerate(lines[1:]):
                line = line.strip()
                line = line.split(' ')
                color, value = line[0], [int(x) for x in line[1:]]

                color_to_index_map[color] = id_
                index_to_value_map[id_] = value

        return color_to_index_map, index_to_value_map

    def convert_target(self, target):
        """
        Convert target segmentation map to [H, W, 1] where the last channel is the index of the class.

        Args:
            target (ndarray): Target segmentation map.

        Returns:
            ndarray: Converted target segmentation map.
        """
        seg_map = np.zeros(target.shape[:2], dtype='uint8')
        for val, id_ in self.value_to_index.items():
            seg_map[(target == list(val)).all(axis=2)] = id_
        return seg_map.reshape(target.shape[0], target.shape[1], 1)

    def get_img_from_path(self, path):
        """
        Load and resize input image from the given path.

        Args:
            path (str): Path to the input image.

        Returns:
            ndarray: Loaded and resized input image.
        """
        image = Image.open(path).convert('RGB')
        image = image.resize(self.img_size)
        return np.array(image)

    def get_target_from_path(self, path):
        """
        Load, resize, and convert target segmentation map from the given path.

        Args:
            path (str): Path to the target segmentation map.

        Returns:
            ndarray: Loaded, resized, and converted target segmentation map.
        """
        img = Image.open(path).convert('RGB')
        img = img.resize(self.img_size)
        img = np.array(img)
        img = self.convert_target(img.astype('uint8'))
        return img

    def prepare_data(self, input_image_dir, target_mask_dir):
        """
        Prepare the input images and target segmentation maps from the given directories.

        Args:
            input_image_dir (str): Directory containing the input images.
            target_mask_dir (str): Directory containing the target segmentation maps.

        Returns:
            tuple: Input images, target segmentation maps.
        """
        #input_image_paths = os.listdir(input_image_dir)
        #target_mask_paths = os.listdir(target_mask_dir)

        num_images = len(input_image_dir)
        input_images = np.zeros((num_images,) + self.img_size + (3,), dtype='float32')
        targets = np.zeros((num_images,) + self.img_size +(1,), dtype="uint8")
        for i in range(num_images):
            input_images[i] = self.get_img_from_path(input_image_dir[i])/255.
            targets[i] = self.get_target_from_path(target_mask_dir[i])
        return input_images, targets