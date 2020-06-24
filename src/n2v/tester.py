import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from datasets import BSD500
from models import UNET_D2
import sys

sys.path.append('..')
import paths


def test():
    global test_dataset
    test_dataset = BSD500(dataset_type=BSD500.TEST)

    # Initializing network
    network = UNET_D2()
    network.to('cpu')
    network.eval()
    instance = '000'
    pretrained_model_folder_path = os.path.join(paths.trained_models_folder_path, 'n2v', 'Instance_' + instance)
    for pretrained_model_file_name in os.listdir(pretrained_model_folder_path)[::-1]:
        try:
            if pretrained_model_file_name.endswith('.pt'):
                network.load_state_dict(
                    torch.load(os.path.join(pretrained_model_folder_path, pretrained_model_file_name)))
                print('Network weights loaded from file: "{}"'.format(pretrained_model_file_name))
            else:
                continue
        except:
            print('Unable to load network weights from file: {}'.format(pretrained_model_file_name))
            continue

        idxes = np.random.randint(0, len(test_dataset), size=15)
        plt.figure(num='Network Performance with weights loaded from file: "{}"'.format(pretrained_model_file_name), figsize=(60, 50))
        plt.suptitle('Noisy Image vs Denoised Image')

        for i in range(15):
            input_image, output_image = test_dataset[idxes[i]]
            predicted_image = network(torch.unsqueeze(input_image, dim=0))[0]
            predicted_image = predicted_image.detach().numpy()
            # predicted_image = (predicted_image - predicted_image.min()) / (predicted_image.max() - predicted_image.min())

            plt.subplot(5, 6, (2 * i) + 1)
            plt.imshow(output_image[0], cmap='gray')
            # plt.title('Noisy Image')

            plt.subplot(5, 6, (2 * i) + 2)
            plt.imshow(predicted_image[0], cmap='gray')
            # plt.title('Denoised Image')

        plt.show()


if __name__ == '__main__':
    print('Commencing Testing')
    test()
    print('Testing Completed')
