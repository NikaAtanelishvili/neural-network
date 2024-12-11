import os.path
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max(z) for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


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

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(a, y):
        return a - y

class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = cost

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
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)

        return a  # Return the pre-activations of the last layer and activations

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, regulation_strength,
                                    test_data=None, monitor_test_accuracy=False, monitor_test_cost=False, monitor_training_accuracy=False, monitor_training_cost=False):

        training_cost, training_accuracy = [], []
        test_cost, test_accuracy = [], []

        for i in range(epochs):
            # if not os.path.exists('model.pkl'):

            random.shuffle(training_data)
            mini_batches = [training_data[j:j + mini_batch_size] for j in
                            range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, regulation_strength, len(training_data))
            print(f'Epoch {i} was completed!')

            if monitor_training_cost:
                cost = self.total_cost(training_data, regulation_strength, vectorized_results=True)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, vectorized_results=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {len(training_data)}")

            if monitor_test_cost:
                cost = self.total_cost(test_data, regulation_strength)
                test_cost.append(cost)
                print(f"Cost on test data: {cost}")

            if monitor_test_accuracy:
                accuracy = self.accuracy(test_data)
                test_accuracy.append(accuracy)
                print(f"Accuracy on test data: {accuracy} / {len(test_data)}")

        return test_cost, test_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, learning_rate, regulation_strength, n):
        # Gradient accumulators
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        for i, (e, l) in enumerate(mini_batch, start=1):
            single_grad_w, single_grad_b = self.back_propagation(e, l)

            # online averaging algorithm
            grad_w = [((i - 1) / i) * gw + (1 / i) * s_gw for gw, s_gw in zip(grad_w, single_grad_w)]
            grad_b = [((i - 1) / i) * gb + (1 / i) * s_gb for gb, s_gb in zip(grad_b, single_grad_b)]

        self.weights = [(1-(regulation_strength * learning_rate)/n)*w - learning_rate * gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - learning_rate * gb for b, gb in zip(self.biases, grad_b)]

    def back_propagation(self, inputs, labels):
        # Initialize gradients for weights and biases
        grad_weights = [np.zeros(w.shape) for w in self.weights]
        grad_biases = [np.zeros(b.shape) for b in self.biases]

        # Forward pass: compute activations and weighted inputs (z-values)
        activation = inputs  # Current layer's activations
        activations = [inputs]  # Store activations for all layers

        z_values = []  # Store weighted input (z-values) for all layers (Wa + b)

        # Replace activation calculation for the output layer:
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weights, activation) + biases
            z_values.append(z)

            # Use softmax for the last layer
            if index == len(self.weights) - 1:
                activation = softmax(z)
            else:
                activation = sigmoid(z)

            activations.append(activation)

        # Compute output layer error (delta)
        delta = self.cost.delta(activations[-1], labels) # BP1

        grad_biases[-1] = delta  # BP3
        grad_weights[-1] = np.dot(delta, activations[-2].T)  # BP4

        # Backpropagate through hidden layers
        for layer in range(2, self.num_layers):
            z_prime = sigmoid_derivative(z_values[-layer])  # Derivative of activation function

            # Compute hidden layer error (current delta)
            delta = np.dot(self.weights[-layer + 1].T, delta) * z_prime

            grad_biases[-layer] = delta  # BP3
            grad_weights[-layer] = np.dot(delta, activations[-layer - 1].T)  # BP4

        return grad_weights, grad_biases

    def accuracy(self, data, vectorized_results=False): #vectorize_result is true for training_data

        if vectorized_results:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)


    def total_cost(self, data, regulation_strength, vectorized_results=False): # vectorized_results are true for training data

        cost = 0.0

        for x, y in data:
            a = self.feedforward(x)

            if not vectorized_results:
                y = vectorize_result(y)

            cost += self.cost.fn(a, y) / len(data)

            # add regulation term
            cost += 0.5 * (regulation_strength / len(data)) * sum(
                np.linalg.norm(w) ** 2 for w in self.weights)

        return cost


def vectorize_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
