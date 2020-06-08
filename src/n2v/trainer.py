import os
import sys
import time
import torch
import json
from torch import optim
from torch import backends
from torch.utils.data import DataLoader
from dataset import Noise2Void
from models import FCNN

sys.path.append('..')
import paths

# --------------------------------------------------------------
# Hyper-parameters
# --------------------------------------------------------------

hyper_paramaters = {
    'PRETRAINED_MODEL_WEIGHTS_FILE_PATH': None,
    'NETWORK': None,
    'MODEL': {
        'BATCH_SIZE': 30,
        'NUM_EPOCHS': 100,
        'NUM_WORKERS': 10
    },
    'OPTIMIZER': {
        'LR': 0.0004,
        'MOMENTUM': 0.9,
        'LOSS_FUNCTION': 'L1'
        # 'LOSS_FUNCTION': 'MSE'
    }
}

if torch.cuda.is_available():
    hyper_paramaters['MODEL']['DEVICE'] = 'cuda'
    torch.cuda.init()
    backends.cudnn.benchmark = True
else:
    hyper_paramaters['MODEL']['DEVICE'] = 'cpu'


# --------------------------------------------------------------

def train():
    # Initializing network
    network = FCNN()

    hyper_paramaters['NETWORK'] = str(network)
    try:
        if hyper_paramaters['PRETRAINED_MODEL_WEIGHTS_FILE_PATH']:
            network.load_state_dict(torch.load(hyper_paramaters['PRETRAINED_MODEL_WEIGHTS_FILE_PATH']))
            print('Network weights initialized from file :"{}"'.format(hyper_paramaters['PRETRAINED_MODEL_WEIGHTS_FILE_PATH']))
    except Exception:
        print('Unable to initialize network weights from given file')
        sys.exit(0)

    # Finding instance folder path
    instance = 0
    while os.path.isdir(os.path.join(paths.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))):
        instance += 1
    instance_folder_path = os.path.join(paths.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))
    os.makedirs(instance_folder_path)
    if not os.path.isdir(paths.trained_models_folder_path):
        os.makedirs(paths.trained_models_folder_path)

    # Saving hyper parameters dictionary
    with open(os.path.join(instance_folder_path, 'hyper_parameters.json'), 'w') as file:
        json.dump(hyper_paramaters, file)

    # Shifting network to appropriate device
    network.to(hyper_paramaters['MODEL']['DEVICE'])
    # Setting network in training mode
    network.train()

    # Initializing dataset
    dataset = Noise2Void(dataset_type=Noise2Void.TRAIN, masking_type=Noise2Void.MASKING_RANDOM_OTHER_PIXEL_VALUE)
    # Defining optimizer
    optimizer = optim.SGD(network.parameters(), lr=hyper_paramaters['OPTIMIZER']['LR'], momentum=hyper_paramaters['OPTIMIZER']['MOMENTUM'])

    loss_function = None
    if hyper_paramaters['OPTIMIZER']['LOSS_FUNCTION'] == 'L1':
        loss_function = torch.nn.L1Loss()
    elif hyper_paramaters['OPTIMIZER']['LOSS_FUNCTION'] == 'MSE':
        loss_function = torch.nn.MSELoss()
    loss_function.to(hyper_paramaters['MODEL']['DEVICE'])

    num_batches = len(dataset) // hyper_paramaters['MODEL']['BATCH_SIZE']
    for epoch_idx in range(hyper_paramaters['MODEL']['NUM_EPOCHS']):
        epoch_start_time = time.time()

        dataset.shuffle()
        train_batcher = DataLoader(dataset=dataset,
                                   batch_size=hyper_paramaters['MODEL']['BATCH_SIZE'],
                                   shuffle=True,
                                   num_workers=hyper_paramaters['MODEL']['NUM_WORKERS'])

        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch_idx + 1, hyper_paramaters['MODEL']['NUM_EPOCHS']))

        epoch_loss = 0

        batch_idx = 0
        for batch in train_batcher:
            print('\tProcessing Batch: {} of {}...'.format(batch_idx + 1, num_batches))
            batch_idx += 1

            input_image, output_image = batch
            input_image = input_image.to(hyper_paramaters['MODEL']['DEVICE'])
            output_image = output_image.to(hyper_paramaters['MODEL']['DEVICE'])

            predicted_image = network(input_image)

            loss = loss_function(predicted_image, output_image)

            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\tBatch (Train) Loss:', loss)
            print()

        torch.save(network.state_dict(), os.path.join(instance_folder_path, 'Model_{}_Epoch_{}.pt'.format(hyper_paramaters['NETWORK'], str(epoch_idx).zfill(3))))
        epoch_end_time = time.time()

        print('Epoch (Train) Loss:', epoch_loss)
        print('Epoch (Train) Time:', epoch_end_time - epoch_start_time)
        print('-' * 80)


if __name__ == '__main__':
    print('Commencing Training')
    train()
    print('Training Completed')
