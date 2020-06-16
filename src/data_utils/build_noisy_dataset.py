import cv2
import os
import numpy as np
import sys

sys.path.append('..')
import paths

IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']
NOISE_PARAMS = {
    'MEAN': 0.0,
    'STD': 0.1
}

for dataset_type in ['train', 'test', 'val']:
    image_idx = 0
    source_dataset_folder_path = os.path.join(paths.bsd_500_original_dataset_folder_path, dataset_type)
    target_dataset_folder_path = os.path.join(paths.bsd_500_noisy_dataset_folder_path, dataset_type)

    if not os.path.isdir(target_dataset_folder_path):
        os.makedirs(target_dataset_folder_path)

    for source_image_file_name in os.listdir(source_dataset_folder_path):
        source_image_file_extension = '.' + source_image_file_name.split('.')[-1].lower()
        if source_image_file_extension in IMAGE_FILE_EXTENSIONS:
            print('Corrupting "{}"'.format(source_image_file_name))
            source_image_file_path = os.path.join(source_dataset_folder_path, source_image_file_name)
            source_image = cv2.imread(source_image_file_path, cv2.IMREAD_COLOR) / 255
            source_image = source_image[:-1, :-1]

            target_image = cv2.add(source_image, np.random.normal(NOISE_PARAMS['MEAN'], NOISE_PARAMS['STD'], size=source_image.shape))
            target_image *= 255

            target_image_file_name = str('{}.jpg'.format(str(image_idx).zfill(3)))
            target_image_file_path = os.path.join(target_dataset_folder_path, target_image_file_name)
            cv2.imwrite(target_image_file_path, target_image)
            print('Noisy file saved!')
            print()
            image_idx += 1
