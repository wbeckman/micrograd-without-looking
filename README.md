# Micrograd - without looking!


This is an attempt to reproduce [Karpathy's micrograd library](https://github.com/karpathy/micrograd/) (which uses reverse-mode autodiff
to calculate gradients) without looking at his source code in order to solidify my understanding
of backpropagation.

It should be noted that I'm doing this purely for pedagogical reasons, and this library is in no
way useful in a production sense.

(If I'm being perfectly honest, I looked once or twice when I got caught up with a backprop bug/to
see the data that he used for his toy example. Sue me.

Specifically, the bug that I encountered was related
to assigning the `backward()` functions on `self._backward` rather than on `result._backward`,
which was causing gradients not to flow backwards.)

## Running

This library requires no external dependencies. I programmed this using Python 3.11.2, but there's
no reason it shouldn't work with other versions of Python. Any version of Python that is backwards
compatible with 3.11.2 should work here as well, and I'm sure plenty of older versions of python3
will run this as well, but I'm not going to attempt to enumerate them here.

In order to train the network to "perfect accuracy" on the 4-record toy dataset that he used,
simply run `python3 main.py` from the root directory and the network will run until it can predict
the outputs perfectly.

## Testing

I added some tests to ensure the functionality of the neural network library and the computational
graph functions. The tests can be run by running `python -m unittest test.py`.
