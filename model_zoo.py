import mxnet as mx
import numpy as np
from layers import *

#Model build from paper: Neural Networks with Few Multiplications
def build_mlp(device,num_epoch,optimizer,lr,use_drelu,model_argument):
    data = mx.symbol.Variable('data')
    if use_drelu:
        fc1 = fully_connected(data, 784, use_drelu=True, activation_label='drelu1')
        fc2 = fully_connected(fc1, 1024, use_drelu=True, activation_label='drelu2')
        fc3 = fully_connected(fc2, 1024, use_drelu=True, activation_label='drelu3')
        fc4 = fully_connected(fc3, 1024, use_drelu=True, activation_label='drelu4')
        fc5 = fully_connected(fc4, 10, activation=False)
        output = softmax(fc5)
    else:
        fc1 = fully_connected(data, 784)
        fc2 = fully_connected(fc1, 1024)
        fc3 = fully_connected(fc2, 1024)
        fc4 = fully_connected(fc3, 1024)
        fc5 = fully_connected(fc4, 10, activation=False)
        output = softmax(fc3)
    return mx.model.FeedForward(
        symbol=output, 
        ctx=device,
        optimizer=optimizer,
        initializer=mx.init.Xavier(factor_type='in',magnitude=2.5),
        num_epoch=num_epoch,
        learning_rate=lr,
        **model_argument)

#Model build from paper: 
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
