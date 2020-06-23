import os
import inspect

current_file_path = inspect.getfile(inspect.currentframe())
project_folder_path = os.path.abspath(os.path.join(current_file_path, '..', '..'))

data_folder_path = os.path.join(project_folder_path, 'data')
src_folder_path = os.path.join(project_folder_path, 'src')
resrc_folder_path = os.path.join(project_folder_path, 'resrc')
trained_models_folder_path = os.path.join(project_folder_path, 'trained models')
results_folder_path = os.path.join(project_folder_path, 'results')

# BSD500 Dataset
bsd_500_dataset_folder_path = os.path.join(data_folder_path, 'BSD500')

bsd_500_original_dataset_folder_path = os.path.join(bsd_500_dataset_folder_path, 'original')
bsd_500_original_train_dataset_folder_path = os.path.join(bsd_500_original_dataset_folder_path, 'train')
bsd_500_original_test_dataset_folder_path = os.path.join(bsd_500_original_dataset_folder_path, 'test')
bsd_500_original_validation_dataset_folder_path = os.path.join(bsd_500_original_dataset_folder_path, 'val')

bsd_500_noisy_dataset_folder_path = os.path.join(bsd_500_dataset_folder_path, 'noisy')
bsd_500_noisy_train_dataset_folder_path = os.path.join(bsd_500_noisy_dataset_folder_path, 'train')
bsd_500_noisy_test_dataset_folder_path = os.path.join(bsd_500_noisy_dataset_folder_path, 'test')
bsd_500_noisy_validation_dataset_folder_path = os.path.join(bsd_500_noisy_dataset_folder_path, 'val')

bsd_500_augmented_dataset_folder_path = os.path.join(bsd_500_dataset_folder_path, 'augmented')
bsd_500_augmented_train_dataset_folder_path = os.path.join(bsd_500_augmented_dataset_folder_path, 'train')
bsd_500_augmented_test_dataset_folder_path = os.path.join(bsd_500_augmented_dataset_folder_path, 'test')
bsd_500_augmented_validation_dataset_folder_path = os.path.join(bsd_500_augmented_dataset_folder_path, 'val')


# CTC Dataset
CTC_dataset_folder_path = os.path.join(data_folder_path, 'CTC')
CTC_original_dataset_folder_path = os.path.join(CTC_dataset_folder_path, 'original')
CTC_few_dataset_folder_path = os.path.join(CTC_dataset_folder_path, 'few')

# Custom Dataset
Custom_dataset_folder_path = os.path.join(data_folder_path, 'Custom')