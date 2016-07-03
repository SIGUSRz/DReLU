import mxnet as mx
import minpy.numpy as np
from minpy.core import grad_and_loss
from layers import *
from initializer import *
import utils
import flags

class mlp:
    def __init__(self,**kwargs):
        self.structure = kwargs.pop('model_structure')
        self.act_mode = kwargs.pop('act_mode', None)
        self.fc_layers = len(self.structure) - 1
        self.context = [mx.gpu(i) for i in range(kwargs['device'])]
        self.record_activation = kwargs.pop('record_activation', False)
        self.record_param = kwargs.pop('record_parameters', False)
        self.param, self.drelu_pos = gaussian_random(structure=self.structure,mode=self.act_mode,weight=kwargs['weight_init'],lower=kwargs['lower_bound'],upper=kwargs['upper_bound'])

        flags.NAME = 'MLP'
        for l in self.structure:
            flags.NAME += '-' + str(l)
        for a in self.act_mode:
            flags.NAME += '-' + a

    def _forward(self, X, *args):
        w0, b0 = args[0], args[1]
        lower1, upper1 = None, None
        if 0 in self.drelu_pos:
            index = 2 * (self.fc_layers + self.drelu_pos.index(0))
            lower1, upper1 = args[index], args[index + 1]
        w1, b1 = args[2], args[3]
        lower2, upper2 = None, None
        if 1 in self.drelu_pos:
            index = 2 * (self.fc_layers + self.drelu_pos.index(1))
            lower2, upper2 = args[index], args[index + 1]
        w2, b2 = args[4], args[5]

        fc1 = fully_connected(X, w0, b0)
        ac1 = activate(fc1, self.act_mode[0], lower1, upper1)
        fc2 = fully_connected(ac1, w1, b1)
        ac2 = activate(fc2, self.act_mode[1], lower2, upper2)
        fc3 = fully_connected(ac2, w2, b2)

        if self.record_activation:
            activation_records = {
                'fc1': fc1.asnumpy(), 
                'ac1': ac1.asnumpy(),
                'fc2': fc2.asnumpy(),
                'ac2': ac2.asnumpy(),
                'fc3': fc3.asnumpy()
            }
            utils.record_activation(activation_records)
        if self.record_param:
            parameter_records = {
                'w0': w0.asnumpy(),
                'b0': b0.asnumpy(),
                'w1': w1.asnumpy(),
                'b1': b1.asnumpy(),
                'w2': w2.asnumpy(),
                'b2': b2.asnumpy(),
                'lower1': lower1.asnumpy() if lower1 != None else None,
                'upper1': upper1.asnumpy() if upper1 != None else None,
                'lower2': lower2.asnumpy() if lower2 != None else None,
                'upper2': upper2.asnumpy() if upper2 != None else None
            }
            utils.record_parameter(parameter_records)
        return fc3

    def _softmax_loss(self, X, y, *args):
        N = X.shape[0]
        scores = self._forward(X, *args)
        scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        prob = scores / np.sum(scores, axis=1, keepdims=True)
        loss = np.sum(-np.log(prob[np.arange(N), y])) / float(N)
        return loss

    def loss(self, X, y=None):
        if y is None:
            return self._forward(X, *self.param)
        else:
            backprop = grad_and_loss(self._softmax_loss, range(2, len(self.param) + 2))
            return backprop(X, y, *self.param)
