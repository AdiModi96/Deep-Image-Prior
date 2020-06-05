import random
from dataloader import NoisyDataLoader
import matplotlib.pyplot as plt

dataset = NoisyDataLoader()
dataset.shuffle()
for i in range(5):

    idx = random.randint(0, len(dataset))

    plt.figure(num='Noisy DataLoader Tester', figsize=(20, 10))
    input_image, output_image = dataset[idx]

    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(input_image, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Output Image')
    plt.imshow(output_image, cmap='gray')
    plt.colorbar()

    plt.show()
