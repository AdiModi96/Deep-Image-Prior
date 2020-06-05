import os
import inspect

current_file_path = inspect.getfile(inspect.currentframe())
project_folder_path = os.path.abspath(os.path.join(current_file_path, '..', '..'))

data_folder_path = os.path.join(project_folder_path, 'data')
src_folder_path = os.path.join(project_folder_path, 'src')
resrc_folder_path = os.path.join(project_folder_path, 'resrc')
trained_models_folder_path = os.path.join(project_folder_path, 'trained models')

bsd_500_dataset_folder_path = os.path.join(data_folder_path, 'BSD500')
bsd_500_train_dataset_folder_path = os.path.join(bsd_500_dataset_folder_path, 'train')
bsd_500_test_dataset_folder_path = os.path.join(bsd_500_dataset_folder_path, 'test')
bsd_500_validation_dataset_folder_path = os.path.join(bsd_500_dataset_folder_path, 'val')

noisy_dataset_folder_path = os.path.join(data_folder_path, 'noisy')
noisy_train_dataset_folder_path = os.path.join(noisy_dataset_folder_path, 'train')
noisy_test_dataset_folder_path = os.path.join(noisy_dataset_folder_path, 'test')
noisy_validation_dataset_folder_path = os.path.join(noisy_dataset_folder_path, 'val')

augmented_dataset_folder_path = os.path.join(data_folder_path, 'augmented')
augmented_train_dataset_folder_path = os.path.join(augmented_dataset_folder_path, 'train')
augmented_test_dataset_folder_path = os.path.join(augmented_dataset_folder_path, 'test')
augmented_validation_dataset_folder_path = os.path.join(augmented_dataset_folder_path, 'val')