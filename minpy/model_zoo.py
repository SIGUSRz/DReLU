import mxnet as mx
import minpy.numpy as np
from minpy.core import grad_and_loss
from layers import *
from initializer import *

class mlp:
    def __init__(self):
        model_struct = [784, 256, 64, 10]
        self.context = mx.gpu(0)
        self.param = [np.copy(p) for p in gaussian_random(hidden_size=model_struct,bias_value=0.1)]
    def _forward(self, X, *args):
        w0,b0 = args[0], args[1]
        w1,b1 = args[2], args[3]
        w2,b2 = args[4], args[5]

        fc1 = fully_connected(X, w0, b0)
        ac1 = activate(fc1, 'relu')
        fc2 = fully_connected(fc1, w1, b1)
        ac2 = activate(fc2, 'relu')
        fc3 = fully_connected(fc2, w2, b2)

        return fc3

    def _softmax_loss(self, X, y, *args):
        scores = self._forward(X, *args)
        scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        probability = scores / np.sum(scores, axis=1, keepdims=True)
        N = X.shape[0]
        loss = -np.sum(np.log(probability[np.arange(N), y])) / N
        return loss
    def loss(self, X, y=None):
        if y is None:
            return self._forward(X, *self.param)
        else:
            backprop = grad_and_loss(self._softmax_loss, range(2,len(self.param)+2))
            return backprop(X, y, *self.param)
