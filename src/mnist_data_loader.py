import gzip
import pickle
import random

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from numpy import dtype


def rotate_image(image, max_angle=15):
    # angle = random.uniform(max_angle, max_angle)
    sfx = random.choice([-1, 1])
    angle = max_angle * sfx
    img = Image.fromarray(np.reshape(image, (28, 28)) * 255)
    rotated_img =  img.rotate(angle, resample=Image.BILINEAR, expand=False)
    return np.array(rotated_img).flatten() / 255.0

def load_mnist_data(file_path, training_chunk_size=1000, test_chunk_size=5000):
    with gzip.open(file_path, 'rb') as f:
        tr_d, _, te_d  = pickle.load(f, encoding='latin1')  # Use 'latin1' for compatibility

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = np.array(list(zip(training_inputs, training_results)), dtype=object)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    # Artifi
    if training_chunk_size >= 50000:
        # Rotate training images
        rotated_images = []
        rotated_labels = []
        for (img, label) in zip(tr_d[0], tr_d[1]):
            rotated_img = rotate_image(img)
            rotated_images.append(np.reshape(rotated_img, (784, 1)))
            rotated_labels.append(vectorized_result(label))

            if len(rotated_images) > training_chunk_size:
                break

        rotated_training_data = np.array(list(zip(rotated_images, rotated_labels)), dtype=object)

        augmented_training_data = np.concatenate((training_data, rotated_training_data), axis=0)

        return augmented_training_data[:training_chunk_size], test_data[:test_chunk_size]

    return training_data[:training_chunk_size], test_data[:test_chunk_size]


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e