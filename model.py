import mxnet as mx
import numpy as np

def get_accuracy(output, label):
    error = np.minimum(1, np.absolute(output-label))
    error = np.sum(error)
    accuracy = 1 - error/output.shape[0]
    return accuracy

def print_epoch(epoch, symbol, arguments, state):
    print 'Epoch ' + str(epoch)

def fully_connected(inputs, hidden_size, activation=True):
    if activation:
        return relu(mx.symbol.FullyConnected(data=inputs, num_hidden=hidden_size)
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

def relu(inputs):
    return mx.symbol.Activation(data=inputs, act_type='relu')

def build_vanilla_nn(use_drelu='False'):
    data = mx.symbol.Variable('data')
    if use_drelu:
        pass
    else:
        fc1 = fully_connected(data,256)
        fc2 = fully_connected(fc1, 64)
        fc3 = fully_connected(fc2, 10, activation=False)
        output = softmax(fc3)

def build_nin(features=None,labels=None,num_train=None,device=mx.cpu(),epoch=100,lr=0.01,optimizer='adam'):
    data = mx.symbol.Variable('data')
    conv1_1 = convLayer(data,filter_size=(5,5),stride=(1,1),pad=(2,2),filters=192)
    conv1_2 = convLayer(conv1_1, filter_size=(1,1), stride=(1,1), pad=(0,0),filters=160)
    conv1_3 = convLayer(conv1_2, filter_size=(1,1), stride=(1,1), pad=(0,0), filters=96)
    pool1 = maxPoolLayer(conv1_3, filter_size=(3,3), stride=(2,2), pad=(0,0))
    drop1 = dropout(pool1, rate=0.5)
    conv2_1 = convLayer(drop1,filter_size=(5,5),stride=(1,1),pad=(2,2),filters=192)
    conv2_2 = convLayer(conv2_1, filter_size=(1,1), stride=(1,1), pad=(0,0),filters=192)
    conv2_3 = convLayer(conv2_2, filter_size=(1,1), stride=(1,1), pad=(0,0), filters=192)
    pool2 = avgPoolLayer(conv2_2, filter_size=(3,3), stride=(2,2), pad=(0,0))
    drop2 = dropout(pool2, rate=0.5)
    conv3_1 = convLayer(drop2,filter_size=(3,3),stride=(1,1),pad=(1,1),filters=192)
    conv3_2 = convLayer(conv3_1, filter_size=(1,1), stride=(1,1), pad=(0,0),filters=192)
    conv3_3 = convLayer(conv3_2, filter_size=(1,1), stride=(1,1), pad=(0,0), filters=10)
    pool3 = avgPoolLayer(conv3_3, filter_size=(8,8), stride=(1,1), pad=(0,0))
    output = softmax(output, name='softmax') 
    model = mx.model.FeedForward.create(output, X=features,y=labels,num_epoch=epoch,ctx=device,learning_rate=lr,optimizer=optimizer,epoch_end_callback=print_epoch)
    return model
