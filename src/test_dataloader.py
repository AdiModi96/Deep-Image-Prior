import random
from torch.utils.data import DataLoader
from dataloader import NoisyDataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 128
dataset = NoisyDataLoader(dataset_type=NoisyDataLoader.VALIDATION,
                          masking_type=NoisyDataLoader.MASKING_FIXED_VALUE,
                          masking_fixed_value=0.5)
dataset.shuffle()
batcher = DataLoader(dataset=dataset,
                     batch_size=BATCH_SIZE)
batch = next(iter(batcher))
instance_idx = random.randint(0, len(batch))
input_image, output_image = batch[0][instance_idx], batch[1][instance_idx]
print(input_image.shape)
plt.figure(num='Noisy DataLoader Tester', figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(input_image[0], cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Output Image')
plt.imshow(output_image[0], cmap='gray')
plt.colorbar()

plt.show()
