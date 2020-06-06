import os
import time
import paths
import json
import torch
from torch import optim
from torch import backends
from torch.utils.data import DataLoader
from dataloader import NoisyDataLoader
from model import UNET

# --------------------------------------------------------------
# Hyper-parameters
# --------------------------------------------------------------
hyper_paramaters = {
    'pretrained_model_file_path': None,
    'MODEL': {
        'BATCH_SIZE': 20,
        'NUM_EPOCHS': 100,
        'NUM_WORKERS': 1
    },
    'OPTIMIZER': {
        'LR': 0.0004,
        'BETAS': (0.9, 0.99),
        'EPSILON': 1e-08,
        'LOSS_FUNCTION': torch.nn.L1Loss()
    }
}

if torch.cuda.is_available():
    hyper_paramaters['MODEL']['DEVICE'] = 'cuda'
    torch.cuda.init()
    backends.cudnn.benchmark = True
else:
    hyper_paramaters['MODEL']['DEVICE'] = 'cpu'
hyper_paramaters['OPTIMIZER']['LOSS_FUNCTION'].to(hyper_paramaters['MODEL']['DEVICE'])
# --------------------------------------------------------------


def train():
    # network = EncoderDecoder()
    network = UNET()
    try:
        if hyper_paramaters['pretrained_model_file_path']:
            network.load_state_dict(torch.load(hyper_paramaters['pretrained_model_file_path']))
            print('Network weights initialized from file :"{}"'.format(os.path.abspath(hyper_paramaters['pretrained_model_file_path'])))
    except Exception:
        print('Unable to initialize network weights from file at:', os.path.abspath(hyper_paramaters['pretrained_model_file_path']))
    network.to(hyper_paramaters['MODEL']['DEVICE'])
    network.train()

    train_dataset = NoisyDataLoader(dataset_type=NoisyDataLoader.TRAIN)
    train_dataset.shuffle()

    train_batcher = DataLoader(dataset=train_dataset,
                               batch_size=hyper_paramaters['MODEL']['BATCH_SIZE'],
                               shuffle=True,
                               num_workers=hyper_paramaters['MODEL']['NUM_WORKERS'])

    optimizer = optim.Adam(network.parameters(),
                           lr=hyper_paramaters['OPTIMIZER']['LR'],
                           betas=hyper_paramaters['OPTIMIZER']['BETAS'],
                           eps=hyper_paramaters['OPTIMIZER']['EPSILON'])

    if not os.path.isdir(paths.trained_models_folder_path):
        os.makedirs(paths.trained_models_folder_path)

    instance = 0
    while os.path.isdir(os.path.join(paths.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))):
        instance += 1
    instance_folder_path = os.path.join(paths.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))
    os.makedirs(instance_folder_path)

    # with open(os.path.join(os.path.join(instance_folder_path, 'Hyper-Parameters.json')), 'w') as file:
    #     json.dump(hyper_paramaters, file)

    num_batches = len(train_dataset) // hyper_paramaters['MODEL']['BATCH_SIZE']
    for epoch in range(hyper_paramaters['MODEL']['NUM_EPOCHS']):

        epoch_start_time = time.time()
        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch + 1, hyper_paramaters['MODEL']['NUM_EPOCHS']))

        epoch_loss = 0
        batch_counter = 1

        for batch in train_batcher:  # Get Batch
            print('\tProcessing Batch: {} of {}...'.format(batch_counter, num_batches))
            batch_counter += 1

            input_image, output_image = batch
            input_image = input_image.to(hyper_paramaters['MODEL']['DEVICE'])
            output_image = output_image.to(hyper_paramaters['MODEL']['DEVICE'])

            denoised_input_patch = network(input_image)  # Pass Batch

            loss = hyper_paramaters['OPTIMIZER']['LOSS_FUNCTION'](denoised_input_patch, output_image)  # Calculate Loss

            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights
            print('\tBatch (Train) Loss:', loss)
            print()

        epoch_end_time = time.time()
        train_dataset.shuffle()
        torch.save(network.state_dict(), os.path.join(instance_folder_path, 'Model_Epoch_{}.pt'.format(str(epoch).zfill(3))))

        print('Epoch (Train) Loss:', epoch_loss)
        print('Epoch (Train) Time:', epoch_end_time - epoch_start_time)
        print('-' * 80)


if __name__ == '__main__':
    print('Commencing Training')
    train()
    print('Training Completed')
