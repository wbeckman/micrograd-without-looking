"""
This is an attempt to reproduce Karpathy's micrograd library
(which uses reverse-mode autodiff to calculate gradients)
without looking at his source code in order to solidify my
understanding of backpropagation.

I'm doing this purely for pedagogical reasons, and this
library is in no way useful in a production sense.
"""

import math


class Value:

    """
    A node in a computation graph. Values have data in the form of an int
    or float, a gradient which is initialized to zero, children (which are
    technically "parents" in a forward graph), and a _backward function
    that describes how to calculate the local gradient.
    """

    def __init__(self, data, grad=0.0, children=(), _backward=lambda: None, _op=''):
        self.data = data
        self.grad = grad
        self._backward = _backward
        self._op = _op
        self.children = children # also could be "parents", but ordering from back to front

    def __repr__(self):
        return f'Value (op: {self._op}):\n  '\
            'Data:{round(self.data, 4)}\n  Grad: {round(self.grad, 2)}'

    def zero_grad(self):
        """Sets gradient for current node back to zero"""
        self.grad = 0.0

    def _topo_sort(self):
        """
        For the DAG of values (defined by children), sort all of the
        children in a way that their parents will be guaranteed to be
        called before them, and thus, their gradients will be computed
        correctly.
        """
        to_process = [self]
        visited = set([self])
        order_to_process = []
        while to_process:
            processing = to_process.pop()
            order_to_process.append(processing)
            for child in processing.children:
                if child not in visited:
                    to_process.append(child)
                    visited.add(child)

        return order_to_process

    def backward(self):
        """
        Because this function is only called in the chain of calls in "backwards"
        after topological sort, it is guaranteed that the downstream gradients
        will be pre-computed.
        """
        self.grad = 1.0
        sorted_nodes = self._topo_sort()
        for node in sorted_nodes:
            node._backward()
            
    def add(self, other):
        """Add two values together and assign local gradient"""
        if isinstance(other, (int, float)):
            other = Value(other)

        def backward():
            """
            Add "routes" gradients from result back to current node.
            """
            self.grad += 1.0 * result.grad
            other.grad += 1.0 * result.grad

        result = Value(self.data + other.data, children=(self,other), _backward=backward, _op='+')
        return result

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __neg__(self):
        return self * -1

    def sub(self, other):
        return self.add(-other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.sub(other)

    def mul(self, other):
        """Multiplies two values together and assign local gradient"""
        if isinstance(other, (int, float)):
            other = Value(other)

        def backward():
            """
            Multiply "swaps" magnitudes from inputs and multiplies upstream grad
            """
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result = Value(self.data * other.data, children=(self,other), _backward=backward, _op='*')
        return result

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def pow(self, power):
        """
        Takes self to power. Does not compute gradient for pow, since strange
        numerical issues can arise with the log operation.
        """
        if isinstance(power, (int, float)):
            power = Value(data=power)
        if not isinstance(power, (Value)):
            print('Exponent for \'pow\' must be int, float, or Value.\nNOTE: If' \
                  ' Value is provided, gradient will NOT be computed for the exponent')
            return

        def backward():
            """
            for f(x)=x^n; f'(x)=(n)*x^(n - 1)
            """
            self.grad += (self.data ** (power.data - 1)) * power.data * result.grad

        result = Value(self.data ** power.data, children=(self,power), _backward=backward, _op='**')
        return result
    
    def __pow__(self, other):
        return self.pow(other)
    
    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            other = Value(data=other)

        def backward():
            """
            for f(x)=n^x; f'(x)=n^x log(n)
            """
            self.grad += (other.data ** self.data) * math.log(other.data) * result.grad
            other.grad += (other.data ** (self.data - 1)) * self.data * result.grad

        result = Value(other.data ** self.data, children=(self,other), _backward=backward, _op='rpow')
        return result

    def div(self, other):
        res = (other ** -1) * self
        res._op = '/'
        return res

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return other.div(self)

    def relu(self):
        """ReLU activation"""
        def backward():
            """Gradient is 1 if self.data is greater than 0"""
            self.grad += 1.0 * result.grad if self.data > 0.0 else 0.0

        result = Value(data=max(0, self.data), children=(self,), _backward=backward, _op='ReLU')
        return result
    
    def tanh(self):
        """tanh activation"""
        def _tanh_fn(x):
            return ((math.e ** x) - (math.e ** -x)) / ((math.e ** x) + (math.e ** -x))
        
        def backward():
            """f(tanh(x))' = 1 - tanh(x)^2"""
            self.grad += (1 - (_tanh_fn(self.data) ** 2)) * result.grad

        result = Value(data=_tanh_fn(self.data), children=(self,), _backward=backward, _op='tanh')
        return result
