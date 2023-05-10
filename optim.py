import abc

class Optimizer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    """
    Implements a very basic version of Stochastic Gradient Descent.
    Given the gradient for a parameter, this algorithm simply steps
    small steps in the negative direction of that gradient.
    """
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, params):
        for param in params:
            # Move parameter away from gradient - i.e. decrease the loss
            param.data -= param.grad * self.lr
