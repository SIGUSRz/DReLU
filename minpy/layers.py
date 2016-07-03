import minpy.numpy as np
def fully_connected(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def activate(inputs, mode, lower, upper):
    if mode == 'relu':
        return np.maximum(0, inputs) 
    elif mode == 'drelu':
        return np.minimum(upper, np.maximum(lower, inputs))
    else:
        print 'Not Supported'
