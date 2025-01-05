import gzip
import math
import pickle

import numpy as np
import theano
import theano.tensor as tensor
from numpy.ma.core import reshape
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import softmax
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d


class Sigmoid:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")
    theano.config.device = 'gpu'
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True.")

def load_mnist_data_shared(file_path=None):
    with gzip.open(file_path, 'rb') as f:
        tr_d, va_d, te_d  = pickle.load(f, encoding='latin1')  # Use 'latin1' for compatibility

        def shared(data):
            shared_inputs = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
            shared_labels = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)

            # .cast is used to cast (convert) a tensor variable or array to a specified data type
            return shared_inputs, tensor.cast(shared_labels, 'int32')

    return shared(tr_d), shared(va_d), shared(te_d)


class ConvolutionalNetwork:
    def __init__(self, layers, mini_batch_size):
        self.test_mb_predictions = None
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        # theano symbolic variables
        self.input = tensor.matrix('input')
        self.label = tensor.ivector('label')

        self.params = [param for layer in self.layers for param in layer.params]

        initial_layer = self.layers[0]
        initial_layer.set_input(self.input, self.input, mini_batch_size) # DOUBLE PASS FOR DROPOUT

        # INITIALIZE REST OF THE NETWORK
        for i in range(1, len(self.layers)):
            prev_layer, layer = self.layers[i-1], self.layers[i]

            layer.set_input(prev_layer.output, prev_layer.output, self.mini_batch_size)

        self.output = layers[-1].output
        self.output_dropout = layers[-1].output_dropout

    def stochastic_gradient_descent(self, training_data, mini_batch_size, learning_rate,
                                    regulation_strength, epoches, test_data=None, validation_data=None):

        training_input, training_label = training_data
        validation_input, validation_label = validation_data
        test_input, test_label = test_data

        # calculate number of mini batches for given dataset
        num_training_batches = len(training_label) / mini_batch_size
        num_validation_batches = len(validation_label) / mini_batch_size
        num_test_batches = len(test_label) / mini_batch_size

        l2_regulation_term = sum([sum(layer.weights**2) for layer in self.layers])

        cost = self.layers[-1].cost(self) + 0.5 * regulation_strength * l2_regulation_term/num_training_batches
        grads = tensor.grad(cost, self.params)
        updates = [(param, param-learning_rate*grad) for grad, param in zip(grads, self.params)]

        mb_i = tensor.lscalar()

        # Function takes mb_i (index of mini-batch) as argument.
        # It returns cost, updates params(weights and biases)
        # and "reassigns" symbolic variables self.x/y which hold mini-batch data itself(images and labels)
        train_mb = theano.function([mb_i], cost, updates=updates,
            givens={
                self.input: training_input[mb_i*mini_batch_size : (mb_i+1)*mini_batch_size],
                self.label: training_label[mb_i*mini_batch_size : (mb_i+1)*mini_batch_size]
            })

        validate_mb_accuracy = theano.function([mb_i], self.layers[-1].accuracy(self.label),
            givens={
                self.input: validation_input[mb_i*mini_batch_size : (mb_i+1)*mini_batch_size],
                self.label: validation_label[mb_i*mini_batch_size : (mb_i+1)*mini_batch_size]
            })

        test_mb_accuracy = theano.function(
            [mb_i], self.layers[-1].accuracy(self.label),
            givens={
                self.input: test_input[mb_i * self.mini_batch_size: (mb_i + 1) * self.mini_batch_size],
                self.label: test_label[mb_i * self.mini_batch_size: (mb_i + 1) * self.mini_batch_size]
            })

        self.test_mb_predictions = theano.function(
            [mb_i], self.layers[-1].output,
            givens={
                self.input: test_input[mb_i * self.mini_batch_size: (mb_i + 1) * self.mini_batch_size]
            })

        best_validation_accuracy = 0.0
        best_iteration = 0
        for epoch in range(epoches):
            for mb_i in range(num_training_batches):

                # 0 * 5000 + 0 || 0 * 5000 + 1 || 0 * 5000 + 4999
                # 1 * 5000 + 0 || 1 * 5000 + 1 || 1 * 5000 + 4999
                iteration = num_validation_batches * epoch + mb_i

                cost = train_mb(mb_i)

                if iteration % 1000 == 0:
                    print(f'Training mini-batch number {iteration}')

                if (iteration + 1) % num_training_batches == 0: # Last mini_batch
                    accuracy = np.mean([validate_mb_accuracy(mb_i) for mb_i in range(num_validation_batches)])
                    print(f'Epoch №{epoch} validation accuracy: {accuracy}')

                    if accuracy >= best_validation_accuracy:
                        best_validation_accuracy = accuracy
                        best_iteration = iteration

                        print(f'Best accuracy up til now: {best_validation_accuracy:.2%}')

                        if test_data is not None:
                            accuracy = np.mean([test_mb_accuracy(mb_i) for mb_i in range(num_test_batches)])
                            print(f'The corresponding test accuracy is {accuracy:.2%}')

        print('Network has finished training!')
        print(f'Best validation accuracy was {best_validation_accuracy:.2%} obtained at iteration №{best_iteration}')




class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation_function=Sigmoid.sigmoid, dropout_rate=0.0):
        self.output_dropout = None
        self.input_dropout = None
        self.prediction = None
        self.output = None
        self.input = None
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function

        self.weights = theano.shared(
            np.asarray(
                np.random.Generator.normal(
                    loc=0.0,
                    scale=np.sqrt(1 / self.input_size),
                    size=(self.input_size, self.output_size)
                ),
                dtype=theano.config.floatX
            ),
            name='weights',
            borrow=True
        )

        self.biases = theano.shared(
            np.asarray(
                np.random.Generator.normal(
                    loc=0.0,
                    scale=np.sqrt(1 / self.input_size),
                    size=(self.input_size, 1)
                ),
                dtype=theano.config.floatX
            ),
            name='biases',
            borrow=True
        )

    def set_input(self, input_data, input_dropout_data, mini_batch_size):
        self.input = input_data.reshape(shape=(mini_batch_size, self.input_size))

        # FOR EVALUATING
        # (1 - self.p_dropout) scales the weights during training to ensure that the outputs are not disproportionately large when dropout is applied.
        self.output = self.activation_function(
            (1-self.dropout_rate)*tensor.dot(self.input, self.weights) + self.biases
        )

        self.prediction = tensor.argmax(self.output, axis=1)

        # FOR TRAINING
        self.input_dropout = dropout_layer(input_dropout_data.reshape(shape=(mini_batch_size, self.input_size)), self.dropout_rate)

        self.output_dropout = self.activation_function(
            tensor.dot(self.input_dropout, self.weights) + self.biases)

    def accuracy(self, label):
        return tensor.mean(tensor.eq(label, self.prediction))


class ConvPoolingLayer:
    def __init__(self, filter_shape, image_shape, pool_size=(2, 2), activation_function=Sigmoid.sigmoid):
        self.output = None
        self.output_dropout = None
        self.input = None
        # filter_shape = (number_of_filters, input_channels, filter_height, filter_width)
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation_function = activation_function
        self.pool_size = pool_size

        # (fan-out) number of output connections
        self.output_connections = filter_shape[0]*np.prod(filter_shape[2:])/np.prod(pool_size)

        self.weights = theano.shared(
            np.asarray(
                np.random.Generator.normal(
                    loc=0.0,
                    scale=np.sqrt(1/self.output_connections),
                    size=filter_shape
                ),
                dtype=theano.config.floatX

            ),
            name='weights',
            borrow=True
        )
        self.biases = theano.shared(
            np.asarray(
                np.random.Generator.normal(
                    loc=0.0,
                    scale=np.sqrt(1/self.output_connections),
                    size=(filter_shape, 1)
                )
            ),
            name='biases',
            borrow=True

        )

    def set_input(self, input_data):
        self.input = input_data.reshape(self.image_shape)
        
        conv_output = conv2d(
            input=self.input,
            filters=self.weights,
            image_shape=self.image_shape,
            filter_shape=self.filter_shape)

        pool_output = pool_2d(input=conv_output,
                              ignore_border=True,
                              ds=self.pool_size)

        self.output = self.activation_function(pool_output + self.biases.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # No dropout in conv layers.

class SoftmaxLayer:
    def __init__(self, input_connections, output_connections, dropout_rate):
        self.output_dropout = None
        self.input_dropout = None
        self.prediction = None
        self.output = None
        self.input = None
        self.input_connections = input_connections
        self.output_connections = output_connections
        self.dropout_rate = dropout_rate

        self.weights = theano.shared(
            np.zeros(
                shape=(input_connections, output_connections),
                dtype=theano.config.floatX
            ),
            name='weights',
            borrow=True)

        self.biases = theano.shared(
            np.zeros(
                shape=(input_connections,),
                dtype=theano.config.floatX
            ),
            name='biases',
            borrow=True
        )


    def set_input(self, input_data, input_dropout_data, mini_batch_size):
        self.input = input_data.reshape(size=(mini_batch_size, self.input_connections))
        self.output = softmax((1-self.dropout_rate)*tensor.dot(self.input, self.weights) + self.biases)
        self.prediction = tensor.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(input_dropout_data.reshape(size=(mini_batch_size, self.input_connections)), self.dropout_rate)
        self.output_dropout = softmax(tensor.dot(self.input_dropout, self.weights) + self.biases)

    def cost(self, net):
        return -tensor.mean(tensor.log(self.output_dropout)[tensor.arange(net.labels.shape[0]), net.labels])

    def accuracy(self, label):
        return tensor.mean(tensor.eq(label, self.prediction))

def dropout_layer(layer, dropout_rate):
    # Set seed for the random numbers
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # Generate a theano RandomStreams
    srng = shared_randomstreams.RandomStreams(rng.randint(999999))

    retain_prob = 1 - dropout_rate

    mask = srng.binomial(n=1, p=retain_prob, size=layer.shape, dtype='floatX')
    return layer * tensor.cast(mask, theano.config.floatX)

