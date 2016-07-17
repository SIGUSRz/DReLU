import minpy.numpy as np
from minpy.array import Number
def fully_connected(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def activate(inputs, mode, lower, upper):
    if mode == 'relu':
        return np.maximum(Number(0.0), inputs) 
    elif mode == 'drelu':
        return np.minimum(upper, np.maximum(lower, inputs))
    else:
        print 'Not Supported'
#def dropout(inputs, rate):
    
