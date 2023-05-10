"""
Functions to gauge model performance
"""

def accuracy(ytrue, ypred):
    """Computes accuracy score between labels and predictions"""
    if len(ytrue) != len(ypred):
        print('ytrue and ypred must be the same shape.')
        return

    count_same = 0

    for ground_truth, pred in zip(ytrue, ypred):
        if ground_truth == pred:
            count_same += 1

    return count_same / len(ytrue)
