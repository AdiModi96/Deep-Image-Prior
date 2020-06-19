import os
import sys
import time
import torch
import cv2
from torch import optim
from torch import backends
from datasets import BSD500
from models import UNET_D4

sys.path.append('..')
import paths

# --------------------------------------------------------------
# Hyper-parameters
# --------------------------------------------------------------

BATCH_SIZE = 40
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    LOSS_FUNCTION.to(DEVICE)
    torch.cuda.init()
    backends.cudnn.benchmark = True


# --------------------------------------------------------------


def train():
    # Initializing network
    network = UNET_D4()
    network.to(DEVICE)
    network.train()

    # Initializing dataset
    dataset = BSD500(image_idx=102)

    # Defining optimizer
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    # Finding instance folder path
    instance = 0
    while os.path.isdir(os.path.join(paths.trained_models_folder_path, 'dip', 'Instance_' + str(instance).zfill(3))):
        instance += 1
    instance_folder_path = os.path.join(paths.trained_models_folder_path, 'dip', 'Instance_' + str(instance).zfill(3))
    os.makedirs(instance_folder_path)
    if not os.path.isdir(paths.trained_models_folder_path):
        os.makedirs(paths.trained_models_folder_path)

    cv2.imwrite(os.path.join(instance_folder_path, '000 - input.png'), dataset[0][0] * 255)
    cv2.imwrite(os.path.join(instance_folder_path, '999 - output.png'), dataset[0][1] * 255)
    dataset.indexing_mode(BSD500.INDEXING_MODE_BATCH)

    model_idx = 0
    loss_threshold = 0.5

    for epoch_idx in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch_idx + 1, NUM_EPOCHS))

        epoch_loss = 0

        batch_idx = 0
        for i in range(len(dataset)):
            print('\tProcessing Batch: {} of {}...'.format(batch_idx + 1, len(dataset)))
            batch_idx += 1

            input_image, output_image = dataset[batch_idx]
            input_image = input_image.to(DEVICE)
            output_image = output_image.to(DEVICE)

            predicted_image = network(input_image)

            loss = torch.nn.MSELoss()(predicted_image, output_image)
            if loss < loss_threshold:
                loss_threshold /= 1.1
                cv2.imwrite(os.path.join(instance_folder_path, '{} - Iterations_{} - Loss_{}.png'.format(str(model_idx).zfill(3),
                                                                                                         epoch_idx * len(dataset) + batch_idx + 1,
                                                                                                         round(loss.item(), 5))),
                            BSD500.channels_last(BSD500.from_torch_to_numpy(predicted_image[0])) * 255)
                # torch.save(network.state_dict(), os.path.join(instance_folder_path, '{} - Loss_{}.pt'.format(str(model_idx).zfill(3), round(loss.item(), 5))))
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
