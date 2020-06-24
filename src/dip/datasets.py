import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append('..')
import paths


class BSD500(Dataset):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    VIRTUAL_DATASET_SIZE = 1000

    INDEXING_MODE_INSTANCE = 0
    INDEXING_MODE_BATCH = 1

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
        if dataset_type == BSD500.TRAIN:
            images_folder_path = os.path.join(paths.bsd_500_noisy_dataset_folder_path, 'train')
        elif dataset_type == BSD500.VALIDATION:
            images_folder_path = os.path.join(paths.bsd_500_noisy_dataset_folder_path, 'val')
        elif dataset_type == BSD500.TEST:
            images_folder_path = os.path.join(paths.bsd_500_noisy_dataset_folder_path, 'test')

        image_file_names = os.listdir(images_folder_path)
        if image_idx == None:
            image_idx = np.random.randint(0, len(image_file_names))

        noisy_image_file_path = os.path.join(images_folder_path, image_file_names[image_idx])
        output_image = cv2.imread(noisy_image_file_path, cv2.IMREAD_COLOR) / 255
        input_image = cv2.GaussianBlur(output_image, ksize=(71, 71), sigmaX=25, sigmaY=25)

        torch_output_image = torch.tensor(BSD500.channels_first(output_image), dtype=torch.float32)
        torch_input_image = torch.tensor(BSD500.channels_first(input_image), dtype=torch.float32)

        self.instance = (input_image, output_image)
        self.batch = (torch.unsqueeze(torch_input_image, dim=0), torch.unsqueeze(torch_output_image, dim=0))

        self.mode = BSD500.INDEXING_MODE_INSTANCE

    def __len__(self):
        return BSD500.VIRTUAL_DATASET_SIZE

    def indexing_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == BSD500.INDEXING_MODE_INSTANCE:
            return self.instance
        else:
            return self.batch


class CTC(Dataset):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    VIRTUAL_DATASET_SIZE = 1000

    INDEXING_MODE_INSTANCE = 0
    INDEXING_MODE_BATCH = 1

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

    def __init__(self, image_idx=None):

        images_folder_path = os.path.join(paths.CTC_few_dataset_folder_path)

        image_file_names = os.listdir(images_folder_path)
        if image_idx == None:
            image_idx = np.random.randint(0, len(image_file_names))

        noisy_image_file_path = os.path.join(images_folder_path, image_file_names[image_idx])
        output_image = np.expand_dims(cv2.imread(noisy_image_file_path, cv2.IMREAD_GRAYSCALE) / 255, axis=2)
        input_image = np.expand_dims(cv2.GaussianBlur(output_image, ksize=(71, 71), sigmaX=25, sigmaY=25), axis=2)

        torch_output_image = torch.tensor(CTC.channels_first(output_image), dtype=torch.float32)
        torch_input_image = torch.tensor(CTC.channels_first(input_image), dtype=torch.float32)

        self.instance = (input_image, output_image)
        self.batch = (torch.unsqueeze(torch_input_image, dim=0), torch.unsqueeze(torch_output_image, dim=0))

        self.mode = CTC.INDEXING_MODE_INSTANCE

    def __len__(self):
        return CTC.VIRTUAL_DATASET_SIZE

    def indexing_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == CTC.INDEXING_MODE_INSTANCE:
            return self.instance
        else:
            return self.batch


class Custom(Dataset):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    VIRTUAL_DATASET_SIZE = 1000

    INDEXING_MODE_INSTANCE = 0
    INDEXING_MODE_BATCH = 1

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

    def __init__(self, image_idx=None):

        images_folder_path = os.path.join(paths.Custom_dataset_folder_path)

        image_file_names = os.listdir(images_folder_path)
        if image_idx == None:
            image_idx = np.random.randint(0, len(image_file_names))

        noisy_image_file_path = os.path.join(images_folder_path, image_file_names[image_idx])
        output_image = cv2.imread(noisy_image_file_path, cv2.IMREAD_COLOR) / 255
        input_image = cv2.GaussianBlur(output_image, ksize=(71, 71), sigmaX=25, sigmaY=25)

        torch_output_image = torch.tensor(Custom.channels_first(output_image), dtype=torch.float32)
        torch_input_image = torch.tensor(Custom.channels_first(input_image), dtype=torch.float32)

        self.instance = (input_image, output_image)
        self.batch = (torch.unsqueeze(torch_input_image, dim=0), torch.unsqueeze(torch_output_image, dim=0))

        self.mode = CTC.INDEXING_MODE_INSTANCE

    def __len__(self):
        return Custom.VIRTUAL_DATASET_SIZE

    def indexing_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == Custom.INDEXING_MODE_INSTANCE:
            return self.instance
        else:
            return self.batch
