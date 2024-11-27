import numpy as np

# THIS WILL BE REPLACED BY ReLu
# a' = σ(Wa + b)
def sigmoid(z):
    return 1/(1+np.exp(-z))


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

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SDG(self, training_data, epochs, batch_size, eta, test_data=None):
        for i in range(epochs):
            np.random.shuffle(training_data)
            batches = [ training_data[j:batch_size] for j in range(0, len(training_data), batch_size)]

            for batch in batches:
                self.update_batch(batch, eta)

            if test_data:
                print(f'Epoch {i} {self.evaluate(test_data)} {len(test_data)}')
            else:
                print(f'Epoch {i} was completed!')

