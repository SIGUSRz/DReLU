import mxnet as mx
import numpy as np
from model_zoo import *
import mxnet_mod as mxm

def build_model(**kwargs):
    if kwargs['model_name'] == 'mlp':
        return build_mlp(**kwargs)
    else:
        print 'Not Supported'
        
def train_model(**kwargs):
    kv = kwargs['kv']
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logger = mxm.build_logger(head)
    evaluation_matrix = ['accuracy']
    if kwargs['show_top_accuracy']:
        for top_k in [5,15,20]:
            evaluation_matrix.append(mx.metric.create('top_k_accuracy', top_k=top_k))
    batch_callback = kwargs['batch_callback']
    if batch_callback == None:
        batch_callback = []
    else:
        if not isinstance(batch_callback,list):
            batch_callback = [batch_callback]
    batch_callback.append(mx.callback.Speedometer(kwargs['batch_size'], 10))
    epoch_callback = epoch_nonsave_callback
    return kwargs['model'].fit(
            X=kwargs['train_data'],
            eval_data=kwargs['eval_data'],
            eval_metric=evaluation_matrix,
            kvstore=kv,
            batch_end_callback=batch_callback,
            epoch_end_callback=epoch_callback)

def epoch_nonsave_callback(epoch, sybol, arg, aux):
    dic = {}
    for key in arg:
        if 'drelu' in key:
            layer = key.split('_')[0]
            bound_value =  arg[key].asnumpy()[0]
            if layer in dic.keys():
                dic[layer].append(bound_value)
            else:
                dic[layer] = [bound_value]
    for key in dic:
        print 'Layer: %s Upper: %f Lower: %f' % (key, max(dic[key]), min(dic[key]))

def parse_args():
    parser = __import__('argparse').ArgumentParser(description='Passing Arguments to Build Model')
    parser.add_argument('--network', type=str,default='mlp',help= 'The Network to Use')
    return parser.parse_args(__import__('sys').argv[len(__import__('sys').argv):])
