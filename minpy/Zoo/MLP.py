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
        if self.fc_layers - 1 != len(self.act_mode):
            print 'Illegal Activation Function Setting'
            return None
        self.context = [mx.gpu(i) for i in range(kwargs['device'])]
        self.param, self.bound_pos = gaussian_random(structure=self.structure,mode=self.act_mode,weight=kwargs['weight_init'],bias=kwargs['bias_init'],lower=kwargs['lower_bound'],upper=kwargs['upper_bound'])

        flags.NAME = 'MLP'
        for l in self.structure:
            flags.NAME += '-' + str(l)
        for a in self.act_mode:
            flags.NAME += '-' + a

    def _forward(self, X, *args):
        w0, b0 = args[0], args[1]
        fc = fully_connected(X, w0, b0)
        if flags.RECORD_FLAG:
            activation_record, parameter_record = {}, {}
            activation_record['fc0'] = fc.asnumpy()
            parameter_record['w0'], parameter_record['b0'] = w0.asnumpy(), b0.asnumpy()
        for i in range(len(self.act_mode)):
            lower, upper = None, None
            if i in self.bound_pos:
                index = 2 * (self.fc_layers + self.bound_pos.index(i))
                lower, upper = args[index], args[index + 1]
            ac = activate(fc, self.act_mode[i], lower, upper)
            w_index = 2 * (i + 1)
            fc = fully_connected(ac, args[w_index], args[w_index + 1])
            if flags.RECORD_FLAG:
                activation_record['ac%d' % i], activation_record['fc%d' % (i + 1)] = ac.asnumpy(), fc.asnumpy()
                parameter_record['lower%d' % i] = lower.asnumpy() if lower is not None else None
                parameter_record['upper%d' % i] = upper.asnumpy() if upper is not None else None
                parameter_record['w%d' % (i + 1)], parameter_record['b%d' % (i + 1)] = args[w_index].asnumpy(), args[w_index + 1].asnumpy()
                utils.record_activation(activation_record)
                utils.record_parameter(parameter_record)
        return fc

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
