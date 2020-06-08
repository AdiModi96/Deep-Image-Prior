import numpy as np
from torch.utils.data import DataLoader
from dataset import Noise2Void
import matplotlib.pyplot as plt


def test_batch(db=None):
    BATCH_SIZE = 128
    batcher = DataLoader(dataset=db, batch_size=BATCH_SIZE)

    batch = next(iter(batcher))

    instance_idxes = np.random.randint(0, len(batch[0]), 15)
    plt.figure(num='Dataset Tester', figsize=(60, 50))

    for i in range(len(instance_idxes)):
        input_image, output_image = batch[0][instance_idxes[i]], batch[1][instance_idxes[i]]

        plt.subplot(6, 5, (2 * i) + 1)
        plt.imshow(input_image[0], cmap='gray')

        plt.subplot(6, 5, (2 * i) + 2)
        plt.imshow(output_image[0], cmap='gray')

    plt.show()

def test_instance(db=None):
    plt.figure(num='Dataset Tester', figsize=(20, 10))

    instance_idx = np.random.randint(0, len(db), 1)[0]
    input_image, output_image = db[instance_idx]

    plt.subplot(1, 2, 1)
    plt.imshow(input_image[0], cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image[0], cmap='gray')

    plt.show()

db = Noise2Void(dataset_type=Noise2Void.VALIDATION, masking_type=Noise2Void.MASKING_FIXED_VALUE, masking_fixed_value=0.5)
db.shuffle()
# test_batch(db)
test_instance(db)
