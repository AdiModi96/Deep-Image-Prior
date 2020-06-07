from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import sys
import paths


class NoisyDataLoader(Dataset):
    IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    IMAGE_SIZE = 160
    PATCH_SIZE = 80
    PIXELS_TO_MASK_PER_PATCH = 80

    MASKING_RANDOM_OTHER_PIXEL_VALUE = 0
    MASKING_UNIFORM_RANDOM_VALUE = 1
    MASKING_MEAN_VALUE = 2
    MASKING_1_MINUS_MEAN_VALUE = 3
    MASKING_FIXED_VALUE = 4

    def __init__(self, dataset_type=VALIDATION, masking_type=MASKING_1_MINUS_MEAN_VALUE, masking_fixed_value=0):

        self.dataset_type = dataset_type

        if dataset_type == NoisyDataLoader.TRAIN:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'train')
        if dataset_type == NoisyDataLoader.TEST:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'test')
        if dataset_type == NoisyDataLoader.VALIDATION:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'val')

        self.masking_type = masking_type
        if self.masking_type == NoisyDataLoader.MASKING_FIXED_VALUE:
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
        output_image = image[i * NoisyDataLoader.IMAGE_SIZE: (i + 1) * NoisyDataLoader.IMAGE_SIZE,
                       j * NoisyDataLoader.IMAGE_SIZE: (j + 1) * NoisyDataLoader.IMAGE_SIZE]

        masked_input_image = output_image.copy()
        for i in range(0, output_image.shape[0], NoisyDataLoader.PATCH_SIZE):
            for j in range(0, output_image.shape[1], NoisyDataLoader.PATCH_SIZE):
                target_idxes = np.asarray([i, j]) + np.random.randint(0, NoisyDataLoader.PATCH_SIZE,
                                                                      size=(NoisyDataLoader.PIXELS_TO_MASK_PER_PATCH, 2))
                if self.masking_type == NoisyDataLoader.MASKING_RANDOM_OTHER_PIXEL_VALUE:
                    source_idxes = np.asarray([i, j]) + np.random.randint(0, NoisyDataLoader.PATCH_SIZE,
                                                                          size=(NoisyDataLoader.PIXELS_TO_MASK_PER_PATCH, 2))
                    masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = output_image[source_idxes[:, 0], source_idxes[:, 1]]
                elif self.masking_type == NoisyDataLoader.MASKING_UNIFORM_RANDOM_VALUE:
                    masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = np.random.uniform(0, 1, size=NoisyDataLoader.PIXELS_TO_MASK_PER_PATCH)
                elif self.masking_type == NoisyDataLoader.MASKING_MEAN_VALUE:
                    masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = np.mean(masked_input_image)
                elif self.masking_type == NoisyDataLoader.MASKING_1_MINUS_MEAN_VALUE:
                    masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = 1 - np.mean(masked_input_image)
                elif self.masking_type == NoisyDataLoader.MASKING_FIXED_VALUE:
                    masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = self.masking_fixed_value

        return np.expand_dims(masked_input_image, axis=0), np.expand_dims(output_image, axis=0)
