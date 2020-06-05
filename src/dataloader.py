import torch
from torch.utils.data import Dataset
from torch import tensor
import cv2
import numpy as np
from typing import List
import os
import sys
import project_paths as pp


class NoisyDataLoader(Dataset):
    IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    PATCH_SIZE = (64, 64)
    PIXELS_TO_MASK_PER_PATCH = 64

    def __init__(self, dataset_type=TEST):

        self.dataset_type = dataset_type

        if dataset_type == NoisyDataLoader.TRAIN:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'train')
        if dataset_type == NoisyDataLoader.TEST:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'test')
        if dataset_type == NoisyDataLoader.VALIDATION:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'val')

        self.num_files = 0
        self.image_file_paths = []
        if os.path.isdir(self.images_folder_path):
            for image_file_name in os.listdir(self.images_folder_path):
                image_file_extension = image_file_name.split('.')[-1]
                if image_file_extension in NoisyDataLoader.IMAGE_FILE_EXTENSIONS:
                    self.image_file_paths.append(os.path.join(self.images_folder_path, image_file_name))
                    self.num_files += 1
        else:
            print("Dataset Path Doesn't Exist!")
            sys.exit(0)

    def __len__(self):
        return 12 * self.num_files

    def __getitem__(self, idx):
        clean_patch = self.clean_patches[int(idx) % self.number_of_clean_patches]

        if self.dataset_type != NoisyDataLoader.TRAIN:
            noisy_image = NoisyDataLoader.add_noise(clean_patch, self.noise_type)
            return (NoisyDataLoader.convert_image_to_model_input(np.asarray(noisy_image / 255, dtype=np.float32)),
                    (NoisyDataLoader.convert_image_to_model_input(np.asarray(clean_patch / 255, dtype=np.float32))))
        else:
            noisy_image_1 = NoisyDataLoader.add_noise(clean_patch, self.noise_type)
            noisy_image_2 = NoisyDataLoader.add_noise(clean_patch, self.noise_type)
            return (NoisyDataLoader.convert_image_to_model_input(np.asarray(noisy_image_1 / 255, dtype=np.float32)),
                    (NoisyDataLoader.convert_image_to_model_input(np.asarray(noisy_image_2 / 255, dtype=np.float32))))

    @staticmethod
    def add_noise(noise_type):

        if noise_type == NoisyDataLoader.GAUSSIAN:
            # Adding Zero Mean Gaussian Noise
            noisy_image = clean_patch + np.random.normal(loc=0.0, scale=25, size=clean_patch.shape)

        elif noise_type == NoisyDataLoader.POISSON:
            # Adding Poisson Noise
            noisy_image = np.random.poisson(clean_patch)

        elif noise_type == NoisyDataLoader.TEXT_OVERLAY:
            noisy_image = NoisyDataLoader.text_noise(clean_patch)

        elif noise_type == NoisyDataLoader.SALT_PEPPER:
            noisy_image = NoisyDataLoader.salt_pepper_noise(clean_patch)

        elif noise_type == NoisyDataLoader.RANDOM_IMPULSE:
            noisy_image = NoisyDataLoader.random_impulse_noise(clean_patch)

        return noisy_image

    @staticmethod
    def random_str(length):
        ascii_chars = [i for i in NoisyDataLoader.ASCII_REDUCE]
        ascii_chars_np = np.array(ascii_chars)
        string = np.random.choice(ascii_chars_np, length)
        return "".join(list(string))

    @staticmethod
    def text_noise(img):
        num_str = np.random.randint(3, 10)
        noise = img.copy()
        x, y = img.shape[0], img.shape[1]

        for i in range(num_str):
            string = NoisyDataLoader.random_str(np.random.randint(3, 20))
            font = np.random.choice(NoisyDataLoader.FONTS)
            line_style = np.random.choice(NoisyDataLoader.LINE_STYLES)
            font_size = np.random.uniform(2, 4)
            col = tuple(np.random.randint(0, 255, 3).astype(np.float64))
            # thickness = np.random.randint(3, 10)
            pos = (np.random.randint(0 - x / 100, x - x / 50), np.random.randint(0 - y / 100, y - y / 25))
            noise = cv2.putText(noise, string, pos, font, font_size, col, 3, line_style)

        return noise

    @staticmethod
    def index_1d_to_2d(i, y):
        return i // y, i % y

    @staticmethod
    def salt_pepper_noise(img, black_ratio=0.05, white_ratio=0.05):

        noise = img.copy()

        total = black_ratio + white_ratio
        x, y = img.shape[0], img.shape[1]

        indexes = np.random.choice(np.arange(x * y), size=int(total * x * y),
                                   replace=False)
        b_indexes = np.random.choice(indexes, size=int(black_ratio * x * y),
                                     replace=False)
        w_indexes = np.random.choice(indexes, size=int(white_ratio * x * y),
                                     replace=False)

        vector_index = np.vectorize(lambda i: NoisyDataLoader.index_1d_to_2d(i, y))

        br, bc = vector_index(b_indexes)
        noise[br, bc] = np.array([0, 0, 0])
        wr, wc = vector_index(w_indexes)
        noise[wr, wc] = np.array([255, 255, 255])

        return noise

    @staticmethod
    def random_colour():
        return np.random.randint(0, 255, 3)

    @staticmethod
    def random_impulse_noise(img, ratio=0.4):
        noise = img.copy()

        x, y = img.shape[0], img.shape[1]

        indexes = np.random.choice(np.arange(x * y), size=int(ratio * x * y),
                                   replace=False)

        vector_index = np.vectorize(lambda i: NoisyDataLoader.index_1d_to_2d(i, y))

        r, c = vector_index(indexes)
        noise[r, c] = np.random.randint(0, 256, noise[r, c].shape)
        # print(noise[wr, wc])

        return noise
