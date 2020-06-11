import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import sys
import torch

sys.path.append('..')
import paths


class DeepImagePrior(Dataset):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    DATASET_SIZE = 1000

    def __init__(self, dataset_type=TRAIN, image_idx=None):

        dataset_type = dataset_type

        images_folder_path = None
        if dataset_type == DeepImagePrior.TRAIN:
            images_folder_path = os.path.join(paths.noisy_dataset_folder_path, 'train')
        elif dataset_type == DeepImagePrior.VALIDATION:
            images_folder_path = os.path.join(paths.noisy_dataset_folder_path, 'val')
        elif dataset_type == DeepImagePrior.TEST:
            images_folder_path = os.path.join(paths.noisy_dataset_folder_path, 'val')

        image_file_names = os.listdir(images_folder_path)
        if image_idx == None:
            image_idx = np.random.randint(0, len(image_file_names))

        self.image_file_path = os.path.join(images_folder_path, image_file_names[image_idx])
        self.output_image = cv2.imread(self.image_file_path, cv2.IMREAD_GRAYSCALE) / 255
        self.input_image = np.random.normal(loc=self.output_image.mean(), scale=0.001, size=self.output_image.shape)
        # self.input_image = np.full_like(self.output_image, fill_value=0.5)

        self.torch_output_tensor = torch.tensor(self.output_image, dtype=torch.float32)
        self.torch_input_tensor = torch.tensor(self.input_image, dtype=torch.float32)

    def __len__(self):
        return DeepImagePrior.DATASET_SIZE

    def get_batch(self):
        return torch.unsqueeze(torch.unsqueeze(self.torch_input_tensor, dim=0), dim=0), torch.unsqueeze(torch.unsqueeze(self.torch_output_tensor, dim=0), dim=0)

    def __getitem__(self, idx):
        return self.input_image, self.output_image
