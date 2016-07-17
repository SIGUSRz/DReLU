import minpy.numpy as np
import utils

def sgd(param, grad, **kwargs):
    for p in range(len(param)):
        param[p] -= kwargs['learning_rate'] * grad[p]

def momentum(param, grad, **kwargs):
    lr = kwargs['learning_rate']
    param_index = kwargs['current_index']
    momentum = kwargs['momentum']
    previous = kwargs['previous']
    
