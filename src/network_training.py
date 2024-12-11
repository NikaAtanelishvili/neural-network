from mnist_data_loader import load_mnist_data
from network import *

# Load the MNIST data
training_data, test_data = load_mnist_data("../../data/mnist.pkl.gz", training_chunk_size=100000, test_chunk_size=10000)

net = Network([784, 30, 10], cost=CrossEntropyCost)

# try:
#     net.load_model("model.pkl")
#     print("Loaded pre-trained model!")
# except FileNotFoundError:
#     print("No pre-trained model found, training from scratch.")

test_cost, test_accuracy, training_cost, training_accuracy = net.stochastic_gradient_descent(training_data, epochs=10, mini_batch_size=10, learning_rate=0.5, regulation_strength=0.01, test_data=test_data,
                                monitor_test_cost=True, monitor_test_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)

# Plotting costs
plt.figure(figsize=(10, 5))
plt.plot(training_cost, label="Training Cost", color="blue", marker="o")
plt.plot(test_cost, label="Test Cost", color="orange", marker="o")
plt.title("Cost vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.legend()
plt.grid(True)
plt.show()

# Plotting accuracies
plt.figure(figsize=(10, 5))
plt.plot(training_accuracy, label="Training Accuracy", color="green", marker="o")
plt.plot(test_accuracy, label="Test Accuracy", color="red", marker="o")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# net.save_model("model.pkl")






