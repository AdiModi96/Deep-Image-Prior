import random

from dataloader import NoisyDataLoader
import matplotlib.pyplot as plt

test_dataloader = NoisyDataLoader(dataset_type=NoisyDataLoader.TEST,
                                  noisy_per_image=10,
                                  noise_type=NoisyDataLoader.TEXT_OVERLAY)

for i in range(5):

    idx = random.randint(0, len(test_dataloader))

    plt.figure(num='Data Loader Tester', figsize=(20, 10))
    image_1, image_2 = test_dataloader[idx]

    plt.subplot(1, 2, 1)
    plt.title('Image 1')
    plt.imshow(NoisyDataLoader.convert_model_output_to_image(image_1), cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Image 2')
    plt.imshow(NoisyDataLoader.convert_model_output_to_image(image_2), cmap='gray')
    plt.colorbar()

    plt.show()
