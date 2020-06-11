import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

sys.path.append('..')
import paths


class Noise2Void(Dataset):
    IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    IMAGE_SIZE = 160
    NUM_PATCHES = 10
    PATCH_SIZE = 60
    PIXELS_TO_MASK_PER_PATCH = 60

    MASKING_RANDOM_OTHER_PIXEL_VALUE = 0
    MASKING_UNIFORM_RANDOM_VALUE = 1
    MASKING_MEAN_VALUE = 2
    MASKING_1_MINUS_MEAN_VALUE = 3
    MASKING_FIXED_VALUE = 4

    def __init__(self, dataset_type=VALIDATION, masking_type=MASKING_1_MINUS_MEAN_VALUE, masking_fixed_value=0):

        self.dataset_type = dataset_type

        if dataset_type == Noise2Void.TRAIN:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'train')
        elif dataset_type == Noise2Void.VALIDATION:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'val')
        elif dataset_type == Noise2Void.TEST:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'test')

        self.masking_type = masking_type
        if self.masking_type == Noise2Void.MASKING_FIXED_VALUE:
            self.masking_fixed_value = masking_fixed_value

        self.num_files = 0
        if os.path.isdir(self.images_folder_path):
            self.image_file_names = os.listdir(self.images_folder_path)
            self.num_files = len(self.image_file_names)
        else:
            print("Dataset Path Doesn't Exist!")
            sys.exit(0)

    def __len__(self):
        return 6 * self.num_files

    def shuffle(self):
        np.random.shuffle(self.image_file_names)

    def __getitem__(self, idx):
        image_file_idx = idx // 6
        patch_idx = idx % 6
        image = np.asarray(
            cv2.imread(os.path.join(self.images_folder_path, self.image_file_names[image_file_idx]), cv2.IMREAD_GRAYSCALE) / 255,
            dtype=np.float32)
        if image.shape[0] < image.shape[1]:
            i, j = patch_idx // 3, patch_idx % 3
        else:
            i, j = patch_idx // 2, patch_idx % 2
        output_image = image[i * Noise2Void.IMAGE_SIZE: (i + 1) * Noise2Void.IMAGE_SIZE,
                       j * Noise2Void.IMAGE_SIZE: (j + 1) * Noise2Void.IMAGE_SIZE]

        masked_input_image = output_image.copy()
        for i in range(Noise2Void.NUM_PATCHES):

            patch_anchor_idx = np.random.randint(0, Noise2Void.IMAGE_SIZE - Noise2Void.PATCH_SIZE, size=2)
            target_idxes = patch_anchor_idx + np.random.randint(0, Noise2Void.PATCH_SIZE, size=(Noise2Void.PIXELS_TO_MASK_PER_PATCH, 2))

            if self.masking_type == Noise2Void.MASKING_RANDOM_OTHER_PIXEL_VALUE:
                source_idxes = patch_anchor_idx + np.random.randint(0, Noise2Void.PATCH_SIZE, size=(Noise2Void.PIXELS_TO_MASK_PER_PATCH, 2))
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = output_image[source_idxes[:, 0], source_idxes[:, 1]]
            elif self.masking_type == Noise2Void.MASKING_UNIFORM_RANDOM_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = np.random.uniform(0, 1, size=Noise2Void.PIXELS_TO_MASK_PER_PATCH)
            elif self.masking_type == Noise2Void.MASKING_MEAN_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = np.mean(masked_input_image)
            elif self.masking_type == Noise2Void.MASKING_1_MINUS_MEAN_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = 1 - np.mean(masked_input_image)
            elif self.masking_type == Noise2Void.MASKING_FIXED_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = self.masking_fixed_value

        return torch.unsqueeze(torch.as_tensor(masked_input_image, dtype=torch.float32), dim=0),\
               torch.unsqueeze(torch.as_tensor(output_image, dtype=torch.float32), dim=0)
