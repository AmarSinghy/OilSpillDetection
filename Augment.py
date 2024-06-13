import os
import cv2
import numpy as np
import albumentations as A

class DataAugmentation:
    def __init__(self, image_dir, mask_dir, output_image_dir, output_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        self.augmentation_techniques = [
            'Normal',
            'HorizontalFlip',
            'Rotate',
            'RandomContrast',
            'RandomBrightness',
            'GaussianBlur'
        ]

        self.augmentation_pipelines = [
            None,  # Normal
            A.HorizontalFlip(p=1),  # HorizontalFlip
            A.Rotate(limit=(-30, 30), p=1),  # Rotate
            A.RandomContrast(limit=(0.5, 1.5), p=1),  # RandomContrast
            A.RandomBrightness(limit=(-0.2, 0.2), p=1),  # RandomBrightness
            A.GaussianBlur(blur_limit=(3, 5), p=1)  # GaussianBlur
        ]

    def apply_augmentation(self, image, mask, augmentation_pipeline):
        if augmentation_pipeline is None:
            return image, mask

        augmented = augmentation_pipeline(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']
        return augmented_image, augmented_mask

    def process_images(self):
        image_files = os.listdir(self.image_dir)
        mask_files = os.listdir(self.mask_dir)

        for image_file in image_files:
            image_path = os.path.join(self.image_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            mask_file = image_file.replace('.jpg', '.png')
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Error loading mask: {mask_path}")
                continue

            # Save the normal image and mask
            normal_image_filename = f'Normal_{image_file}'
            normal_mask_filename = f'Normal_{mask_file}'

            normal_image_path = os.path.join(self.output_image_dir, normal_image_filename)
            normal_mask_path = os.path.join(self.output_mask_dir, normal_mask_filename)

            cv2.imwrite(normal_image_path, image)
            cv2.imwrite(normal_mask_path, mask)

            print(f"Saved normal image: {normal_image_path}")
            print(f"Saved normal mask : {normal_mask_path}")
            
            for technique, pipeline in zip(self.augmentation_techniques[1:], self.augmentation_pipelines[1:]):
                augmented_image, augmented_mask = self.apply_augmentation(image, mask, pipeline)

                augmented_image_filename = f'{technique}_{image_file}'
                augmented_mask_filename = f'{technique}_{mask_file}'

                augmented_image_path = os.path.join(self.output_image_dir, augmented_image_filename)
                augmented_mask_path = os.path.join(self.output_mask_dir, augmented_mask_filename)
  
                cv2.imwrite(augmented_image_path, augmented_image)
                cv2.imwrite(augmented_mask_path, augmented_mask)

                print(f"Saved augmented image: {augmented_image_path}")
                print(f"Saved augmented mask: {augmented_mask_path}")
