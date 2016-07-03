import minpy.numpy as np
import utils

def sgd(param, grad, **kwargs):
    for p in range(len(param)):
        param[p] -= kwargs['learning_rate'] * grad[p]
