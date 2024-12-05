import os.path
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt

# THIS WILL BE REPLACED BY ReLu
# a' = σ(Wa + b)
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(activation, l):
    """Return the vector of partial derivatives"""
    # C is mean squared error
    return activation - l


def visualize_incorrect_predictions(incorrect_predictions, max_images=10):
    if len(incorrect_predictions) == 0: return
    num_images = min(len(incorrect_predictions), max_images)
    fig, axes = plt.subplots(1, num_images, figsize=(10, 5))
    fig.suptitle("Incorrect Predictions", fontsize=16)

    for i, (pred, image, label) in enumerate(incorrect_predictions[:num_images]):
        # Reshape the image to 28x28 for visualization
        img = image.reshape(28, 28)

        # Display the image
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {pred}\nTrue: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


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

    def load_model(self, filename):
        with open(filename, "rb") as f:
            model_data = pickle.load(f)
        self.weights = model_data["weights"]
        self.biases = model_data["biases"]

    def save_model(self, filename):
        print('model saved')
        model_data = {
            "weights": self.weights,
            "biases": self.biases
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

    def feedforward(self, a):
        z_values = []  # To store pre-activations for softmax
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_values.append(z)
            a = sigmoid(z)

        return z_values[-1], a  # Return the pre-activations of the last layer and activations

    def sdg(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        for i in range(epochs):
            if not os.path.exists('model.pkl'):
                random.shuffle(training_data)
                mini_batches = [ training_data[j:j+mini_batch_size] for j in range(0, len(training_data), mini_batch_size)]

                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                print(f'Epoch {i} {self.evaluate(test_data)} {len(test_data)}')
            else:
                print(f'Epoch {i} was completed!')

    def update_mini_batch(self, mini_batch, eta):
        # Gradient accumulators
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        for i, (e, l) in enumerate(mini_batch, start=1):
            single_grad_w, single_grad_b = self.back_propagation(e, l)

            # online averaging algorithm
            grad_w = [((i-1)/i) * gw + (1 / i) * s_gw for gw, s_gw in zip(grad_w, single_grad_w)]
            grad_b = [((i-1)/i) * gb + (1 / i) * s_gb for gb, s_gb in zip(grad_b, single_grad_b)]

        self.weights = [w - eta * gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - eta * gb for b, gb in zip(self.biases, grad_b)]

    def back_propagation(self, inputs, labels):
        """
        Perform backpropagation to calculate the gradients of the cost function
        with respect to the weights and biases of the neural network.

        Args:
            inputs (np.ndarray): Input data e.g. activations from the first layer.
            labels (np.ndarray): Expected outputs.

        Returns:
            tuple: A tuple containing two lists:
                - grad_weights: Gradients of the cost function with respect to weights.
                - grad_biases: Gradients of the cost function with respect to biases.
        """

        # Initialize gradients for weights and biases
        grad_weights = [np.zeros(w.shape) for w in self.weights]
        grad_biases = [np.zeros(b.shape) for b in self.biases]

        # Forward pass: compute activations and weighted inputs (z-values)
        activation = inputs  # Current layer's activations
        activations = [inputs]  # Store activations for all layers

        z_values = []  # Store weighted input (z-values) for all layers (Wa + b)

        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weights, activation) + biases
            z_values.append(z)

            if index == len(self.weights) - 1:  # Last layer
                # Apply softmax for the last layer
                activation = np.exp(z) / np.sum(np.exp(z), axis=0)
            else:
                # Apply sigmoid for hidden layers
                activation = sigmoid(z)

            activations.append(activation)

        # Compute output layer error (delta)
        delta = cost_derivative(activations[-1], labels) # BP1

        grad_biases[-1] = delta  # BP3
        grad_weights[-1] = np.dot(delta, activations[-2].T)  # BP4

        # Backpropagate through hidden layers
        for layer in range(2, self.num_layers):
            z_prime = sigmoid_derivative(z_values[-layer])  # Derivative of activation function

            # Compute hidden layer error (current delta)
            delta = np.dot(self.weights[-layer + 1].T, delta) * z_prime

            grad_biases[-layer] = delta # BP3
            grad_weights[-layer] = np.dot(delta, activations[-layer - 1].T) # BP4

        return grad_weights, grad_biases

    def evaluate(self, test_data):
        incorrect_predictions = []

        test_results = []
        for (e, l) in test_data:
            z, res = self.feedforward(e)

            probability_distribution = np.exp(z) / np.sum(np.exp(z), axis=0)

            print(np.array_str(probability_distribution, precision=4, suppress_small=True))

            pred = np.argmax(res)

            test_results.append((pred, l))
            if pred != l:
                incorrect_predictions.append((pred, e, l))

        visualize_incorrect_predictions(incorrect_predictions)

        return sum(int(pred == ans) for (pred, ans) in test_results)

