import minpy.numpy as np
import flags
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
def dropout(inputs, rate):
    if flags.MODE == "Train":
        mask = np.random.random(inputs.shape) > rate
        return inputs * mask
    else:
        return inputs
def convolution(inputs, filter_size, stride, pad, filters):
    convLayer = mx.symbol.Convolution(data=inputs, kernel=filter_size, stride=stride, pad=pad, num_filter=filters)
    return convLayer 

def maxPoolLayer(inputs, filter_size, stride, pad):
    return mx.symbol.Pooling(data=inputs, kernel=filter_size, stride=stride, pad=pad, pool_type='max')

def avgPoolLayer(inputs, filter_size, stride, pad):
    return mx.symbol.Pooling(data=inputs, kernel=filter_size, stride=stride, pad=pad, pool_type='avg')


