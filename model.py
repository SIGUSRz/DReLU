import mxnet as mx
import numpy as np
from layers import *
import log
import argparse
import sys

def build_model(model_name='mlp',device=mx.cpu(),num_epoch=20,batch_size=128,initializer='Xavier',optimizer='sgd',lr=0.01,use_drelu='False'):
    model_argument = {}
    model_argument['epoch_size'] = 60000 / batch_size
    if model_name  == 'mlp':
        return build_mlp(device,num_epoch,optimizer,lr,use_drelu,model_argument)
    else:
        print 'no'
        
def train_model(net, train_data=(None,None),eval_data=(None,None),batch_size=128,batch_callback=None,kvstore='local',save_model_path=None,top_accuracy=1):
    kv = mx.kvstore.create(kvstore)
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logger = log.build_logger()
    evaluation_matrix = ['accuracy']
    if top_accuracy > 1:
        evaluation_matrix.append(mx.metric.create('top_k_accuracy', top_k=top_accuracy))
    if batch_callback == None:
        batch_callback = []
    else:
        batch_callback.append(mx.callback.Speedometer(batch_size,50))
        print batch_callback
    if save_model_path == None:
        epoch_callback = epoch_nonsave_callback
    else:
        pass
    return net.fit(
            X=train_data[0],
            y=train_data[1],
            eval_data=eval_data,
            kvstore=kv,
            batch_end_callback=batch_callback,
            epoch_end_callback=epoch_callback)

def epoch_nonsave_callback(epoch, sybol, arg, aux):
    dic = {}
    #print arg
    for key in arg:
        if 'activation' in key:
            if key.split('_')[0] in dic:
                d[key.split('_')[0]].append(arg[key].asnumpy())
            else:
                d[key.split('_')[0]] = [arg[key].asnumpy()[0]]
    for key in dic:
        print '%s %f %f' % (key, d[key][0], d[key][1])

def build_mlp(device,num_epoch,optimizer,lr,use_drelu,model_argument):
    data = mx.symbol.Variable('data')
    if use_drelu:
        fc1 = fully_connected(data,256)
        fc2 = fully_connected(fc1, 64)
        fc3 = fully_connected(fc2, 10, activation=False)
        output = softmax(fc3)
    else:
        fc1 = fully_connected(data,256)
        fc2 = fully_connected(fc1, 64)
        fc3 = fully_connected(fc2, 10, activation=False)
        output = softmax(fc3)
    return mx.model.FeedForward(
        symbol=output, 
        ctx=device,
        optimizer=optimizer,
        initializer=mx.init.Xavier(factor_type='in',magnitude=2.34),
        num_epoch=num_epoch,
        learning_rate=lr,
        **model_argument)
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
def parse_args():
    parser = argparse.ArgumentParser(description='Passing Arguments to Build Model')
    parser.add_argument('--network', type=str,default='mlp',help= 'The Network to Use')
    return parser.parse_args(sys.argv[len(sys.argv):])
