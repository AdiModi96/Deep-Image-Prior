import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append('..')
import paths


class BSD500(Dataset):
    IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    IMAGE_SIZE = 160
    NUM_PATCHES = 5
    PATCH_SIZE = 60
    PIXELS_TO_MASK_PER_PATCH = 60

    MASKING_RANDOM_OTHER_PIXEL_VALUE = 0
    MASKING_UNIFORM_RANDOM_VALUE = 1
    MASKING_MEAN_VALUE = 2
    MASKING_1_MINUS_MEAN_VALUE = 3
    MASKING_FIXED_VALUE = 4

    @staticmethod
    def channels_first(image):
        return np.transpose(image, (2, 0, 1))

    @staticmethod
    def channels_last(image):
        return np.transpose(image, (1, 2, 0))

    @staticmethod
    def from_torch_to_numpy(tensor):
        return tensor.to('cpu').detach().numpy()

    @staticmethod
    def from_numpy_to_torch(tensor):
        return torch.tensor(tensor)

    def __init__(self, dataset_type=VALIDATION, masking_type=MASKING_1_MINUS_MEAN_VALUE, masking_fixed_value=0):

        self.dataset_type = dataset_type

        if dataset_type == BSD500.TRAIN:
            self.images_folder_path = os.path.join(paths.bsd_500_augmented_train_dataset_folder_path)
        elif dataset_type == BSD500.VALIDATION:
            self.images_folder_path = os.path.join(paths.bsd_500_augmented_validation_dataset_folder_path)
        elif dataset_type == BSD500.TEST:
            self.images_folder_path = os.path.join(paths.bsd_500_augmented_test_dataset_folder_path)

        self.masking_type = masking_type
        if self.masking_type == BSD500.MASKING_FIXED_VALUE:
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
        image = cv2.cvtColor(cv2.imread(os.path.join(self.images_folder_path, self.image_file_names[image_file_idx]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) /  255
        if image.shape[0] < image.shape[1]:
            i, j = patch_idx // 3, patch_idx % 3
        else:
            i, j = patch_idx // 2, patch_idx % 2
        output_image = image[i * BSD500.IMAGE_SIZE: (i + 1) * BSD500.IMAGE_SIZE, j * BSD500.IMAGE_SIZE: (j + 1) * BSD500.IMAGE_SIZE]

        masked_input_image = output_image.copy()
        for i in range(BSD500.NUM_PATCHES):

            patch_anchor_idx = np.random.randint(0, BSD500.IMAGE_SIZE - BSD500.PATCH_SIZE, size=2)
            target_idxes = patch_anchor_idx + np.random.randint(0, BSD500.PATCH_SIZE, size=(BSD500.PIXELS_TO_MASK_PER_PATCH, 2))

            if self.masking_type == BSD500.MASKING_RANDOM_OTHER_PIXEL_VALUE:
                source_idxes = patch_anchor_idx + np.random.randint(0, BSD500.PATCH_SIZE, size=(BSD500.PIXELS_TO_MASK_PER_PATCH, 2))
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = output_image[source_idxes[:, 0], source_idxes[:, 1]]
            elif self.masking_type == BSD500.MASKING_UNIFORM_RANDOM_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = np.random.uniform(0, 1, size=BSD500.PIXELS_TO_MASK_PER_PATCH)
            elif self.masking_type == BSD500.MASKING_MEAN_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = np.mean(masked_input_image)
            elif self.masking_type == BSD500.MASKING_1_MINUS_MEAN_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = 1 - np.mean(masked_input_image)
            elif self.masking_type == BSD500.MASKING_FIXED_VALUE:
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = self.masking_fixed_value

        return torch.as_tensor(BSD500.channels_first(output_image), dtype=torch.float), torch.as_tensor(BSD500.channels_first(masked_input_image), dtype=torch.float)
