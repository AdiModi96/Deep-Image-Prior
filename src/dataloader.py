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

    def __init__(self, dataset_type=TEST):

        self.dataset_type = dataset_type

        if dataset_type == NoisyDataLoader.TRAIN:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'train')
        if dataset_type == NoisyDataLoader.TEST:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'test')
        if dataset_type == NoisyDataLoader.VALIDATION:
            self.images_folder_path = os.path.join(paths.augmented_dataset_folder_path, 'val')

        self.num_files = 0
        self.image_file_paths = []
        if os.path.isdir(self.images_folder_path):
            for image_file_name in os.listdir(self.images_folder_path):
                image_file_extension = '.' + image_file_name.split('.')[-1].lower()
                if image_file_extension in NoisyDataLoader.IMAGE_FILE_EXTENSIONS:
                    self.image_file_paths.append(os.path.join(self.images_folder_path, image_file_name))
                    self.num_files += 1
        else:
            print("Dataset Path Doesn't Exist!")
            sys.exit(0)

    def __len__(self):
        return 6 * self.num_files

    def shuffle(self):
        np.random.shuffle(self.image_file_paths)

    def __getitem__(self, idx):
        file_idx = idx // 6
        patch_idx = idx % 6
        image = np.asarray(cv2.imread(self.image_file_paths[file_idx], cv2.IMREAD_GRAYSCALE) / 255, dtype=np.float32)
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
                source_idxes = np.asarray([i, j]) + np.random.randint(0, NoisyDataLoader.PATCH_SIZE,
                                                                      size=(NoisyDataLoader.PIXELS_TO_MASK_PER_PATCH, 2))
                masked_input_image[target_idxes[:, 0], target_idxes[:, 1]] = output_image[source_idxes[:, 0], source_idxes[:, 1]]

        return np.expand_dims(masked_input_image, axis=0), np.expand_dims(output_image, axis=0)
