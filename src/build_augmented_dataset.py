import cv2
import project_paths as pp
import os


IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

def augment_from_image(image):
    augmented_images = {}
    rotated_image = None
    for i in range(4):
        if i > 0:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated_image = image

        mirrored_rotated_image = rotated_image.copy()
        augmented_images['{}_{}'.format(i, 0)] = mirrored_rotated_image
        mirrored_rotated_image = cv2.flip(rotated_image, flipCode=1)
        augmented_images['{}_{}'.format(i, 1)] = mirrored_rotated_image

    return augmented_images


source_dataset_folder_path = os.path.join(pp.bsd_500_test_dataset_folder_path)
target_dataset_folder_path = os.path.join(pp.augmented_test_dataset_folder_path)

if not os.path.exists(target_dataset_folder_path):
    os.makedirs(target_dataset_folder_path)

for source_image_file_name in os.listdir(source_dataset_folder_path):
    source_image_file_extension = '.' + source_image_file_name.split('.')[-1].lower()
    if source_image_file_extension in IMAGE_FILE_EXTENSIONS:
        print('Augmenting from', source_image_file_name)
        source_image_file_path = os.path.join(source_dataset_folder_path, source_image_file_name)
        source_image = cv2.imread(source_image_file_path, cv2.IMREAD_COLOR)
        augmented_target_images = augment_from_image(source_image)
        for key in augmented_target_images.keys():
            target_image_file_name = source_image_file_name.replace(source_image_file_extension, '_{}{}'.format(key, source_image_file_extension))
            target_image_file_path = os.path.join(target_dataset_folder_path, target_image_file_name)
            cv2.imwrite(target_image_file_path, augmented_target_images[key])
        print('Files Saved!')
        print()
        # break
import cv2
import project_paths as pp
import os


IMAGE_FILE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

def augment_from_image(image):
    augmented_images = {}
    rotated_image = None
    for i in range(4):
        if i > 0:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated_image = image

        mirrored_rotated_image = rotated_image.copy()
        augmented_images['{}_{}'.format(i, 0)] = mirrored_rotated_image
        mirrored_rotated_image = cv2.flip(rotated_image, flipCode=1)
        augmented_images['{}_{}'.format(i, 1)] = mirrored_rotated_image

    return augmented_images


source_dataset_folder_path = os.path.join(pp.bsd_500_test_dataset_folder_path)
target_dataset_folder_path = os.path.join(pp.augmented_test_dataset_folder_path)

if not os.path.exists(target_dataset_folder_path):
    os.makedirs(target_dataset_folder_path)

for source_image_file_name in os.listdir(source_dataset_folder_path):
    source_image_file_extension = '.' + source_image_file_name.split('.')[-1].lower()
    if source_image_file_extension in IMAGE_FILE_EXTENSIONS:
        print('Augmenting from', source_image_file_name)
        source_image_file_path = os.path.join(source_dataset_folder_path, source_image_file_name)
        source_image = cv2.imread(source_image_file_path, cv2.IMREAD_COLOR)
        augmented_target_images = augment_from_image(source_image)
        for key in augmented_target_images.keys():
            target_image_file_name = source_image_file_name.replace(source_image_file_extension, '_{}{}'.format(key, source_image_file_extension))
            target_image_file_path = os.path.join(target_dataset_folder_path, target_image_file_name)
            cv2.imwrite(target_image_file_path, augmented_target_images[key])
        print('Files Saved!')
        print()
        # break
