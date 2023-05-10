import copy
import random
from nn import NeuralNetwork
from optim import SGD
from criteria import accuracy


def get_dummy_data():
    """Same data Andrej used in his video"""
    X = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    y = [1.0, -1.0, -1.0, 1.0]
    return X, y


if __name__ == '__main__':
    lr = 0.1
    random.seed(48)
    X, y = get_dummy_data()
    nn_input_dim = 3
    nn_output_dims = [4, 4, 1]
    optim = SGD(lr=lr)
    network = NeuralNetwork(input_shape=nn_input_dim,
                            output_shapes=nn_output_dims, optim=optim)
    preds = [round(network.forward(rec)[0].data) for rec in X]
    accuracy_score = accuracy(y, preds)
    print(f"Accuracy before: {accuracy_score}")
    new_params = network.params()

    # Iterate over data by epoch until accuracy is perfect
    while accuracy_score != 1.0:
        preds = []
        losses = []
        old_params = copy.deepcopy(new_params)
        for idx, record in enumerate(X):
            label = y[idx]
            pred = network.forward(record)
            loss = network.compute_loss(pred, label)
            losses.append(loss.data)
            loss.backward()
            network.update()
            network.zero_grad()
            y_pred = round(pred[0].data)
            preds.append(y_pred)

        new_params = network.params()
        accuracy_score = accuracy(y, preds)
        diff = sum([abs(new.data - old.data) for new, old in zip(new_params, old_params)])
        print('EPOCH STATS:')
        print(f'  Avg. loss: {sum(losses) / len(losses)}')
        print(f'  Accuracy: {accuracy_score}')
        print(f'  Total change in all params: {diff}')

    preds = [round(network.forward(rec)[0].data) for rec in X]
    accuracy_post = accuracy(y, preds)
    print(f"Final accuracy: {accuracy_post}")
