import numpy as np

class Network:
    def __init__(self, sizes):
        """
            Initialize the neural network.

            Parameters:
            sizes (list of int): A list where each element represents the number
                                 of neurons in the corresponding layer of the
                                 neural network. The length of the list determines
                                 the number of layers in the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

