import matplotlib.pyplot as plt
from datasets import BSD500


def test_instance(db=None):
    db.indexing_mode(BSD500.INDEXING_MODE_INSTANCE)
    input_image, output_image = db[0]

    plt.figure(num='Dataset Tester', figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Output Image')

    plt.show()


db = BSD500(dataset_type=BSD500.TRAIN)
test_instance(db)
