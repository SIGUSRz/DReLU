import minpy.numpy as np

def fully_connected(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def activate(inputs, mode):
    if mode == 'relu':
        lower = np.copy(np.zeros(inputs.shape))
        return np.maximum(lower, inputs)
    elif mode == 'drelu':
        upper = np.copy(np.zeros(inputs.shape))
        lower = np.copy(np.zeros(inputs.shape))
        return np.minimum(upper, np.maximum(upper, inputs))
