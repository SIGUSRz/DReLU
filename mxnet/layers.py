import mxnet as mx
import numpy as np
import drelu

def get_accuracy(output, label):
    error = np.minimum(1, np.absolute(output-label))
    error = np.sum(error)
    accuracy = 1 - error/output.shape[0]
    return accuracy

def print_epoch(epoch, symbol, arguments, state):
    print 'Epoch ' + str(epoch)

def activate(inputs,mode,number):
    if mode == 'drelu':
        label = 'drelu' + str(number)
        return drelu.drelu(name=label)(data=inputs,name=label)
    return mx.symbol.Activation(data=inputs, act_type=mode)

def fully_connected(inputs, hidden_size, activation=True,act_func=None,number=None):
    if activation:
        return activate(mx.symbol.FullyConnected(data=inputs, num_hidden=hidden_size),act_func,number)
    else:
        return mx.symbol.FullyConnected(data=inputs, num_hidden=hidden_size)

def convLayer(inputs, filter_size, stride, pad, filters):
    conv = mx.symbol.Convolution(data=inputs, kernel=filter_size, stride=stride, pad=pad, num_filter=filters)
    layer = mx.symbol.Activation(data=conv, act_type='relu')
    return layer

def maxPoolLayer(inputs, filter_size, stride, pad):
    return mx.symbol.Pooling(data=inputs, kernel=filter_size, stride=stride, pad=pad, pool_type='max')

def avgPoolLayer(inputs, filter_size, stride, pad):
    return mx.symbol.Pooling(data=inputs, kernel=filter_size, stride=stride, pad=pad, pool_type='avg')

def dropout(inputs, rate):
    return mx.symbol.Dropout(data=inputs,p=rate)

def softmax(inputs):
    return mx.symbol.SoftmaxOutput(data=inputs, name='softmax')
