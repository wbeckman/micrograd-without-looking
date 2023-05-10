import unittest
import math
import random
from engine import Value
from nn import Neuron, Layer, NeuralNetwork
from criteria import accuracy
from optim import SGD


class TestValueMethods(unittest.TestCase):

    def test_add_forward_variations(self):
        num = Value(10.5)
        float_num_to_add = 5.5
        num_to_add = Value(float_num_to_add)
        expected = 16.0
        assert(num.add(num_to_add).data == expected)
        assert((num + num_to_add).data == expected)
        assert((num + float_num_to_add).data == expected)
        assert((float_num_to_add + num).data == expected)
   
    def test_add_gradient(self):
        num = Value(10.5)
        num_to_add = Value(5.5)
        result = num.add(num_to_add)
        assert(num_to_add.grad == 0.0)
        assert(num.grad == 0.0)
        result.backward()
        assert(num_to_add.grad == 1.0)
        assert(num.grad == 1.0)

    def test_sub_forward_variations(self):
        num = Value(10.5)
        float_num_to_sub = 5.5
        num_to_sub = Value(float_num_to_sub)
        expected = 5.0
        assert(num.sub(num_to_sub).data == expected)
        assert((num - num_to_sub).data == expected)
        assert((num - float_num_to_sub).data == expected)
        assert((float_num_to_sub - num).data == expected)
        
    def test_sub_gradient(self):
        num = Value(10.5)
        num_to_sub = Value(5.5)
        result = num.sub(num_to_sub)
        assert(num_to_sub.grad == 0.0)
        assert(num.grad == 0.0)
        result.backward()
        assert(num_to_sub.grad == -1)
        assert(num.grad == 1)

    def test_mul_forward_variations(self):
        float_multiplier = 2.0
        num = Value(10.5)
        multiplier = Value(float_multiplier)
        expected = 21.0
        assert((num.mul(multiplier)).data == expected)
        assert((num * multiplier).data == expected)
        assert((num * float_multiplier).data == expected)
        assert((float_multiplier * num).data == expected)

    def test_mul_gradient(self):
        num = Value(10.5)
        multiplier = Value(2.0)
        result = num.mul(multiplier)
        assert(num.grad == 0.0)
        assert(multiplier.grad == 0.0)
        result.backward()
        assert(num.grad == 2.0)
        assert(multiplier.grad == 10.5)

    def test_negate_forward(self):
        num = Value(10.5)
        assert(-num.data == -10.5)
    
    def test_negate_gradient(self):
        original_val = Value(10.5)
        assert(original_val.grad == 0.0)
        result = -original_val
        result.backward()
        assert(original_val.grad == -1.0)

    def test_div_forward_variations(self):
        denom_float = 2.0
        num = Value(10.0)
        denom = Value(denom_float)
        expected = 5.0
        reverse_expected = 0.2
        assert((num / denom).data == expected)
        assert((num / denom_float).data == expected)
        assert((num.div(denom)).data == expected)
        assert((denom / num).data == reverse_expected)
        assert((denom_float / num).data == reverse_expected)
        assert((denom.div(num)).data == reverse_expected)

    def test_div_backward(self):
        num = Value(10.0)
        denom = Value(2.0)
        result = (num / denom)
        assert(num.grad == 0.0)
        assert(denom.grad == 0.0)
        result.backward()
        assert(num.grad == 1/2)
        assert(denom.grad == (-10/4))

    def test_pow_forward_variations(self):
        power_float = 2.0
        num = Value(10.0)
        power = Value(power_float)
        expected = 100
        assert((num ** power).data == expected)
        assert((num.pow(power)).data == expected)
        assert((num ** power_float).data == expected)

    def test_pow_backward(self):
        num = Value(10.0)
        power = Value(2.0)
        result = num ** power
        assert(num.grad == 0.0)
        result.backward()
        assert(num.grad == 20.0)

    def test_rpow_forward(self):
        num = Value(5.0)
        power_float = 2.0
        result = power_float ** num
        assert(result.data == 32.0)

    def test_rpow_backward(self):
        num = Value(5.0)
        power_float = 2.0
        result = power_float ** num
        result.backward()
        expected_grad = math.log(power_float) * (power_float**num.data)
        assert(num.grad == expected_grad)

    def test_relu_forward(self):
        num_lt_zero = Value(-10.0)
        num_gt_zero = Value(10.0)
        result_lt_zero = num_lt_zero.relu()
        result_gt_zero = num_gt_zero.relu()
        assert(result_lt_zero.data == 0.0)
        assert(result_gt_zero.data == num_gt_zero.data)

    def test_relu_backward(self):
        num_lt_zero = Value(-10.0)
        num_gt_zero = Value(10.0)
        result_lt_zero = num_lt_zero.relu()
        result_gt_zero = num_gt_zero.relu()
        assert(num_lt_zero.grad == 0.0)
        assert(num_gt_zero.grad == 0.0)
        result_lt_zero.backward()
        result_gt_zero.backward()
        assert(num_lt_zero.grad == 0.0)
        assert(num_gt_zero.grad == 1.0)

    def test_tanh_forward(self):
        input_ = Value(3.0)
        res = input_.tanh()
        expected = 0.99505475
        assert(round(res.data, 8) == expected)

    def test_tanh_backward(self):
        input_ = Value(3.0)
        res = input_.tanh()
        res.backward()
        expected = 0.00986604
        assert(round(input_.grad, 8) == expected)


class TestNNMethods(unittest.TestCase):

    def setUp(self):
        random.seed(41)
        self.n = Neuron(2)
        self.testing_scalar = 2.0
        self.layer = Layer(5, 5, activation='relu')
        self.network = NeuralNetwork(5, [5, 2], None)
        self.input_ = [self.testing_scalar] * 5

    def test_neuron_forward(self):
        output = self.n.forward(self.input_)
        expected = sum([2 * param for param in self.n.params()])
        assert(output.data == expected.data)

    def test_neuron_backward(self):
        output = self.n.forward(self.input_)
        output.backward()
        for param in self.n.params():
            assert(param.grad == self.testing_scalar)

    def test_layer_forward(self):
        output = self.layer.forward(self.input_)
        
        for idx, neuron in enumerate(self.layer.neurons):
            res = 0
            for param in neuron.params():
                res += self.testing_scalar * param.data # sum(wx)
            res += self.layer.bias.data # + b
            res = max(0, res) # ReLU
            assert(res == output[idx].data)
            
    
    def test_layer_backward(self):
        _ = self.layer.forward(self.input_)
        active_activations = [n for n in self.layer.neurons if n.output.data > 0]
        inactive_activations = [n for n in self.layer.neurons if n.output.data <= 0]
        for n in inactive_activations:
            assert(all([param.grad == 0.0 for param in n.params()]))
        for n in active_activations:
            assert(all([param.grad == 0.0 for param in n.params()]))

        n_active_neurons = len(active_activations)
        self.layer.backward()

        for n in inactive_activations:
            assert(all([param.grad == 0.0 for param in n.params()]))
        for n in active_activations:
            assert(all([param.grad == 2.0 for param in n.params()]))

        assert(self.layer.bias.grad == n_active_neurons)

    def test_nn_grad_prop(self):
        # Pretty weak test just to see if any gradients are propagated to first layer
        preds = self.network.forward(self.input_)
        loss = self.network.compute_loss(preds, 1)
        loss.backward()
        grad_sum = 0.0
        for w in self.network.layers[0].params():
            grad_sum += w.grad
        
        assert grad_sum != 0.0


class TestCriteria(unittest.TestCase):
    
    def test_accuracy(self):
        pred = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        actual=[1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        assert(accuracy(actual, pred) == 0.7)
    
    def test_accuracy_failure_different_lengths(self):
        pred = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        actual = [1, 0, 0, 0, 0, 0, 1, 0, 0]
        assert(accuracy(actual, pred) == None)


class TestOptim(unittest.TestCase):

    def test_sgd(self):
        sgd_optim = SGD(lr=0.01)
        val1 = Value(100.0) # Grad = 10.0
        val2 = Value(10.0) # Grad = 100.0
        val3 = val1 * val2 # Grad = 1.0
        val3.backward() 
        sgd_optim.step([val1, val2, val3])
        assert(val1.data == 99.9)
        assert(val2.data == 9.0)
        assert(val3.data == 999.99)
        val1.zero_grad()
        val2.zero_grad()
        val3.zero_grad()
        sgd_optim.step([val1, val2, val3])
        assert(val1.data == 99.9)
        assert(val2.data == 9.0)
        assert(val3.data == 999.99)


if __name__ == '__main__':
    unittest.main()
