import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import sys

sys.path.append('..')
import paths


def root_square_error(image_1, image_2):
    return np.sqrt((image_1 - image_2) ** 2)


def init_animation():
    global images, image_idx, image_file_names, predicted_image_sub_plot, error_sub_plot
    plt.subplot(2, 2, 1)
    plt.title('Input Image')
    plt.imshow(images[0], cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Output Image')
    plt.imshow(images[-1], cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('')
    predicted_image_sub_plot = plt.imshow(np.zeros_like(images[0]), cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('')
    error_sub_plot = plt.imshow(np.zeros_like(images[0]), cmap='gray')


def animate(index):
    global images, image_idx, image_file_names, predicted_image_sub_plot, error_sub_plot

    plt.subplot(2, 2, 3)
    plt.title('Predicted Image: {}'.format(image_file_names[index]))
    predicted_image_sub_plot.set_data(images[index])

    plt.subplot(2, 2, 4)
    plt.title('Root Square Error Image')
    error_sub_plot.set_data(root_square_error(images[index], images[-1]))


for i in range(45, 55):
    instance = str(i).zfill(3)
    print(instance)
    images_folder_path = os.path.join(paths.results_folder_path, 'dip', 'Instance_' + instance)
    if not os.path.isdir(images_folder_path):
        print('Images folder path Doesn\'t exist')
        print('Quitting')

    images = []
    image_file_names = []
    for file_name in os.listdir(images_folder_path):
        if file_name.endswith('.png'):
            image_file_names.append(file_name)
            image = cv2.cvtColor(cv2.imread(os.path.join(images_folder_path, file_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255
            images.append(image)

    figure = plt.figure(num='Instance_' + instance, figsize=(20, 20))

    predicted_image_sub_plot = None
    error_sub_plot = None

    anim = animation.FuncAnimation(fig=figure, func=animate, frames=len(images), interval=1000, init_func=init_animation)
    # plt.show()
    anim.save(os.path.join(images_folder_path, 'animation.mp4'), writer=animation.FFMpegWriter(fps=2))
