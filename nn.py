"""
Classes used to compose a neural network. Defines a set of values
for a neuron, a list of neurons to be used in a layer, and then
a list of layers to be used in the network. The network
backpropagates errors and update weight sets with an optimizer.
"""

import random
import functools
import abc
from engine import Value


class Module(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, input_):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass


class Neuron(Module):
    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1,1), _op='weight') for _ in range(n_inputs)]
        self.output = None

    def forward(self, X):
        outputs = []
        for w, i in zip(self.weights, X):
            outputs.append(w * i)
        self.output = functools.reduce(lambda x,y: x+y, outputs)
        return self.output

    def params(self):
        return self.weights
    
    def backward(self):
        if not self.output:
            print('Forward pass for neuron has not been completed')
            return
        self.output.backward()
    
    def zero_grad(self):
        for w in self.weights:
            w.zero_grad()


class Layer(Module):
    """
    Layer in a neural network. Creates layer_size neurons as output nodes
    from an input of size n_inputs. 
    """
    def __init__(self, layer_size, n_inputs, activation='tanh', add_bias=True):
        self.neurons = [Neuron(n_inputs) for _ in range(layer_size)]
        self.add_bias = add_bias
        self.bias = Value(random.uniform(-1,1), _op='bias')
        self.output = None

        if activation.lower() == 'relu':
            self.activation = self._relu

        if activation.lower() == 'tanh':
            self.activation = self._tanh
        
        if activation.lower() == 'none':
            self.activation = self._identity
        
    def _relu(self, input_):
        return [i.relu() for i in input_]
    
    def _tanh(self, input_):
        return [i.tanh() for i in input_]
    
    def _identity(self, input_):
        return input_

    def forward(self, input_):
        outputs = []
        for neuron in self.neurons:
            # Layer output is activation_fn(wX + b)
            if self.add_bias:
                out = neuron.forward(input_) + self.bias
            else:
                out = neuron.forward(input_)
            outputs.append(out)
        
        self.output = self.activation(outputs)
        return self.output
    
    def backward(self):
        if not self.output:
            print('Forward pass for layer has not been completed')
            return

        for output in self.output:
            output.backward()

    def params(self):
        params = []
        for neuron in self.neurons:
            params += neuron.params()
        return params + [self.bias]

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

        self.bias.zero_grad()

class NeuralNetwork(Module):

    """
    Fairly inflexible, simple multi-layer perceptron using tanh activation
    functions and mean-squared error for outputs.
    """

    def __init__(self, input_shape, output_shapes, optim):
        self.layers = []
        
        for output_shape in output_shapes:
            layer = Layer(output_shape, input_shape, activation='tanh')
            input_shape = output_shape
            self.layers.append(layer)

        self.output = None
        self.optim = optim
    
    def compute_loss(self, predictions, true_output):
        """
        Currently just mean-squared error for binary classification
        """
        return ((predictions[0] - true_output) ** 2)

    def forward(self, input_):
        for layer in self.layers:
            fwd = layer.forward(input_)
            input_ = fwd

        self.output = fwd
        return self.output

    def update(self):
        self.optim.step(self.params())

    def params(self):
        output = []
        for layer in self.layers:
            output += layer.params()
        return output
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
