import numpy as np
import typing

import utils
import mnist

np.random.seed(1)


def get_mean_and_std(verbose: bool):
    '''
    Using entire training dataset to find the mean pixel value and the standard deviation
    '''
    # Get all training images in the dataset to ensure these values never change
    images, _, _, _ = mnist.load()
    # calculate mean and standard deviation with numpy
    mean = np.mean(images)
    std = np.std(images)
    if verbose:
        print(f'''
        number of images   = {len(images)}
        mean pixel value   = {mean}
        standard deviation = {std}
        ''')
    return mean, std


def pre_process_images(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    '''
    Args:
        X (np.ndarray): images of shape [batch size, 784] in the range (0, 255)
        verbose (bool, optional): wether to print the mean and std values used in normalization. Defaults to False.
    Returns:
        np.ndarray: images of shape [batch size, 785] normalized as described in task2a
    '''
    # Normalize Values
    mean_pixel_value, standard_deviation = get_mean_and_std(verbose)
    X = (X - mean_pixel_value) / standard_deviation
    # Bias Trick
    bias = np.ones([X.shape[0], 1])
    return np.hstack([X, bias])


def one_hot_encode(Y: np.ndarray, num_classes: int):
    '''
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    '''
    return np.arange(num_classes) == Y


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    '''
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        float: Cross entropy error
    '''
    return np.sum(-np.sum((targets * np.log(outputs))))/outputs.shape[0]


class SoftmaxModel:

    def __init__(self,
                 neurons_per_layer: typing.List[int],
                 use_improved_weight_init: bool = False,
                 use_improved_sigmoid: bool = False,
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I: int = 785
        # Find number of hidden layers
        self.num_hidden_layers: int = len(neurons_per_layer) - 1
        # Initialize the weights
        self.weights: typing.List[np.ndarray] = []
        num_inputs = self.I
        for num_outputs in neurons_per_layer:
            shape = (num_inputs, num_outputs)
            if use_improved_weight_init:
                # randomized weights from normal distribution
                mean, std = 0, 1 / np.sqrt(num_inputs)
                weights = np.random.normal(mean, std, shape)
            else:
                # uniformly randomized weights
                weights = np.random.uniform(-1, 1, shape)
            self.weights.append(weights)
            num_inputs = num_outputs
        # Set improved or not
        self.use_improved_sigmoid: bool = use_improved_sigmoid
        # Set all grads to None
        self.zero_grad()
        # Initialize arrays for storing activations from forward pass
        self.pre_activations = [None] * self.num_hidden_layers
        self.activations = [None] * self.num_hidden_layers

        print(f'''\nInitialized Model With {self.num_hidden_layers} Hidden Layer(s):
            Inputs:          {self.I}
            Hidden Layer(s): {', '.join([str(n) for n in neurons_per_layer[:-1]])}
            Outputs:         {neurons_per_layer[-1]}''')

    def zero_grad(self) -> None:
        self.grads = [None] * len(self.weights)

    def forward(self, X: np.ndarray) -> np.ndarray:
        ''' Applies Sigmoid to all layers except the last one which uses SoftMax
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        '''
        for layer in range(self.num_hidden_layers):
            # Activations (output) from previous layer
            prev_activation = self.activations[layer - 1] if layer != 0 else X
            # Weights to this layer
            weight = self.weights[layer]

            pre_activation = np.dot(prev_activation, weight)
            self.pre_activations[layer] = pre_activation
            # Apply sigmoid as activation function
            activation = self.sigmoid(pre_activation)
            self.activations[layer] = activation

        # Find activation of the layer before the output layer
        last_layer_activation = self.activations[-1] if self.num_hidden_layers > 0 else X
        # Weights to output layer
        weight_to_output = self.weights[-1]

        output_pre_activation = np.dot(last_layer_activation, weight_to_output)
        # Return Softmax of the output
        return self.softmax(output_pre_activation)

    def backward(self,
                 X: np.ndarray,
                 outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        '''
        Computes the gradients and saves it to the variable self.grads

        Args:
            X: images of shape [batch size, 785]    
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        '''
        # Start from the back (Output Layer)
        delta = -(targets - outputs)
        activation = self.activations[-1].T  # of previous layer
        self.grads[-1] = activation.dot(delta)

        # Hidden Layers In Reverse Order
        for layer in reversed(range(self.num_hidden_layers)):
            # Activation (output) of the previous layer (transposed)
            activation = (self.activations[layer-1] if layer != 0 else X).T
            # Pre Activation of the current layer
            pre_activation = self.pre_activations[layer]
            # Weights to the next layer (transposed)
            weight = self.weights[layer+1].T

            delta = delta.dot(weight) * self.sigmoid_derivative(pre_activation)
            self.grads[layer] = activation.dot(delta)

        # Divide all grads by the batch size
        batch_size = X.shape[0]
        self.grads = [grad/batch_size for grad in self.grads]

    def softmax(self, y_hat: np.array) -> np.array:
        ''' Return Softmax '''
        return np.exp(y_hat) / np.sum(np.exp(y_hat), axis=1, keepdims=True)

    def sigmoid(self, z: np.array) -> np.array:
        ''' Return normal or improved sigmoid '''
        if self.use_improved_sigmoid:
            return 1.7159 * np.tanh(2/3 * z)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z: np.array) -> np.array:
        ''' Return normal or improved sigmoid derivative '''
        if self.use_improved_sigmoid:
            return 1.7159 * 2/3 * (1 - (np.tanh(2/3 * z))**2)
        return self.sigmoid(z) * (1 - self.sigmoid(z))
