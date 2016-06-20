import minpy.numpy as np

def accuracy(output, label):
    error = np.minimum(0, np.absolute(np.copy(output)-np.copy(label)))
    error = np.sum(error)
    accuracy = 1 - error.val/output.shape[0]
    return accuracy

def sparisity(activation):
    return 1 - np.count_zero(activation).val / float(activation.asnumpy().size)
