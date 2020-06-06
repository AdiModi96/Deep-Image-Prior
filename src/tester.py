import torch
from model import UNET
import os
import paths as pp
from matplotlib import pyplot as plt
from dataloader import NoisyDataLoader
import random


def test():
    global test_dataset
    test_dataset = NoisyDataLoader(dataset_type=NoisyDataLoader.TEST)

    # Initializing network
    network = UNET()
    network.to('cpu')
    instance = '000'
    pretrained_model_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + instance)
    for pretrained_model_file_name in os.listdir(pretrained_model_folder_path):
        try:
            if pretrained_model_file_name.endswith('.pt'):
                network.load_state_dict(
                    torch.load(os.path.join(pretrained_model_folder_path, pretrained_model_file_name)))
                print('Network weights initialized using file from:', pretrained_model_file_name)
            else:
                continue
        except:
            print('Unable to load network with weights from:', pretrained_model_file_name)
            continue

        idx = random.randint(0, len(test_dataset))
        input_image, output_image = test_dataset[idx]
        predicted_image = network(torch.unsqueeze(torch.as_tensor(input_image), dim=0))[0]
        predicted_image = predicted_image.detach().numpy()

        plt.figure(num='Network Performance using weights at {}'.format(pretrained_model_file_name), figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(output_image[0], cmap='gray')
        plt.colorbar()
        plt.title('Noisy Image')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_image[0], cmap='gray')
        plt.colorbar()
        plt.title('Denoised Image')

        plt.show()


if __name__ == '__main__':
    print('Commencing Testing')
    test()
    print('Testing Completed')
