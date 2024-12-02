from network import Network
from mnist_data_loader import load_mnist_data

# Load the MNIST data
training_data, test_data = load_mnist_data("../../data/mnist.pkl.gz", chunk_size=200)

net = Network([784, 30, 10])

try:
    net.load_model("model.pkl")
    print("Loaded pre-trained model!")
except FileNotFoundError:
    print("No pre-trained model found, training from scratch.")

# Train the model
net.sdg(training_data, epochs=2, mini_batch_size=10, eta=0.1, test_data=test_data)

# Save the trained model
net.save_model("model.pkl")




