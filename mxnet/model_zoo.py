import mxnet as mx
import numpy as np
import mxnet_mod as mxm
from layers import *

#Model build from paper: Neural Networks with Few Multiplications
def build_mlp(**kwargs):
    data = mx.symbol.Variable('data')
    device = kwargs['device']
    num_epoch = kwargs['num_epoch']
    optimizer = kwargs['optimizer']
    lr = kwargs['lr']
    act_func = kwargs['act_func']

    fc1 = fully_connected(data, 784, act_func=act_func[0], number=0)
    fc2 = fully_connected(fc1, 1024, act_func=act_func[1], number=1)
    fc3 = fully_connected(fc2, 1024, act_func=act_func[2], number=2)
    fc4 = fully_connected(fc3, 1024, act_func=act_func[3], number=3)
    fc5 = fully_connected(fc4, 10, activation=False, number=4)
    output = softmax(fc5)
    
    return mx.model.FeedForward(
        symbol=output, 
        ctx=device,
        optimizer=optimizer,
        initializer=mxm.Custom_Initializer(**kwargs),
        num_epoch=num_epoch,
        learning_rate=lr)

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
    model = mx.model.FeedForward(
        symbol=output, 
        ctx=device,
        optimizer=optimizer,
        initializer=mx.init.Xavier(rnd_type=kwargs['rnd_type'],factor_type=kwargs['factor_type'],magnitude=kwargs['magnitude']),
        num_epoch=epoch,learning_rate=lr,epoch_end_callback=print_epoch)
    return model
