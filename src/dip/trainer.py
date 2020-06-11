import os
import sys
import time
import torch
import cv2
import json
from torch import optim
from torch import backends
from dataset import DeepImagePrior
from matplotlib import pyplot as plt
from models import UNET_Heavy

sys.path.append('..')
import paths

# --------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------

pretrained_model_file_path = None

MODEL = {
    'NUM_EPOCHS': 100,
}
if torch.cuda.is_available():
    MODEL['DEVICE'] = 'cuda'
    torch.cuda.init()
    torch.backends.cudnn.benchmark = True
else:
    MODEL['DEVICE'] = 'cpu'

OPTIMIZER = {
    'LR': 0.0001,
    'BETAS': (0.9, 0.99),
    'EPSILON': 1e-08,
    'LOSS_FUNCTION': torch.nn.MSELoss().to(MODEL['DEVICE'])
}


# --------------------------------------------------------------

def train():
    # Initializing network
    network = UNET_Heavy()
    network.to(MODEL['DEVICE'])

    try:
        if pretrained_model_file_path:
            network.load_state_dict(torch.load(pretrained_model_file_path))
            print('Network weights initialized from file :"{}"'.format(pretrained_model_file_path))
    except Exception:
        print('Unable to initialize network weights from given file')
        sys.exit(0)

    # Finding instance folder path
    instance = 0
    while os.path.isdir(os.path.join(paths.trained_models_folder_path, 'dip', 'Instance_' + str(instance).zfill(3))):
        instance += 1
    instance_folder_path = os.path.join(paths.trained_models_folder_path, 'dip', 'Instance_' + str(instance).zfill(3))
    os.makedirs(instance_folder_path)
    if not os.path.isdir(paths.trained_models_folder_path):
        os.makedirs(paths.trained_models_folder_path)

    # print('Model Summary:')
    # print(summary(model=network, input_size=(1, 160, 160)))
    # Setting network in training mode
    network.train()

    # Initializing dataset
    dataset = DeepImagePrior(dataset_type=DeepImagePrior.TRAIN, image_idx=165)
    # Defining optimizer

    optimizer = optim.Adam(network.parameters(),
                           lr=OPTIMIZER['LR'],
                           betas=OPTIMIZER['BETAS'],
                           eps=OPTIMIZER['EPSILON'])

    model_idx = 0
    loss_threshold = 0.5

    cv2.imwrite(os.path.join(instance_folder_path, '.input.png'), dataset[0][0] * 255)
    cv2.imwrite(os.path.join(instance_folder_path, '.output.png'), dataset[0][1] * 255)

    for epoch_idx in range(MODEL['NUM_EPOCHS']):
        epoch_start_time = time.time()

        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch_idx + 1, MODEL['NUM_EPOCHS']))

        epoch_loss = 0

        batch_idx = 0
        for i in range(len(dataset)):
            print('\tProcessing Batch: {} of {}...'.format(batch_idx + 1, len(dataset)))
            batch_idx += 1

            input_image, output_image = dataset.get_batch()
            input_image = input_image.to(MODEL['DEVICE'])
            output_image = output_image.to(MODEL['DEVICE'])

            predicted_image = network(input_image)

            loss = torch.nn.MSELoss()(predicted_image, output_image)
            if loss < loss_threshold:
                loss_threshold /= 2
                cv2.imwrite(os.path.join(instance_folder_path, 'Model_{}_Loss_{}.png'.format(str(model_idx).zfill(5), round(loss.item(), 5))), predicted_image[0, 0].cpu().detach().numpy() * 255)
                torch.save(network.state_dict(), os.path.join(instance_folder_path, 'Model_{}_Loss_{}.pt'.format(str(model_idx).zfill(5), round(loss.item(), 5))))
                model_idx += 1

            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\tBatch (Train) Loss:', loss)
            print()

        epoch_end_time = time.time()

        print('Epoch (Train) Loss:', epoch_loss)
        print('Epoch (Train) Time:', epoch_end_time - epoch_start_time)
        print('-' * 80)


if __name__ == '__main__':
    print('Commencing Training')
    train()
    print('Training Completed')
