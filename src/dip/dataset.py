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
    VIRTUAL_DATASET_SIZE = 1000

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
        self.output_image = cv2.imread(self.image_file_path, cv2.IMREAD_COLOR) / 255
        self.input_image = cv2.GaussianBlur(self.output_image, ksize=(51, 51), sigmaX=15, sigmaY=15)

        self.torch_output_image = torch.tensor(DeepImagePrior.channels_first(self.output_image), dtype=torch.float32)
        self.torch_input_image = torch.tensor(DeepImagePrior.channels_first(self.input_image), dtype=torch.float32)

        self.batch = (torch.unsqueeze(self.torch_input_image, dim=0), torch.unsqueeze(self.torch_output_image, dim=0))

    def __len__(self):
        return DeepImagePrior.VIRTUAL_DATASET_SIZE

    def get_batch(self):
        return self.batch

    def __getitem__(self, idx):
        return self.input_image, self.output_image
