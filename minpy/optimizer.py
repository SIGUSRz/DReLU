import minpy.numpy as np
import utils

def sgd(param, grad, **kwargs):
    return param - kwargs['learning_rate'] * grad

def momentum(param, grad, **kwargs):
    lr = kwargs['learning_rate']
    momentum = kwargs['momentum']
    previous = kwargs['previous']
    previous_param_index = kwargs['current_index']
    return param - (lr * grad + momentum * previous[previous_param_index])
