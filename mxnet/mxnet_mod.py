import mxnet as mx
import logging
import sys

def build_logger(head):
    reload(logging)
    logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    return logger

class Custom_Initializer(mx.initializer.Initializer):
    def __init__(self, **kwargs):
        super(Custom_Initializer,self).__init__()
        if kwargs['use_drelu']:
            self.upper_bound = kwargs['upper_bound']
            self.lower_bound = kwargs['lower_bound']
        self.method = kwargs['initializer']
        if self.method == 'xavier':
            self.rnd_type = kwargs['rnd_type']
            self.factor_type = kwargs['factor_type']
            self.magnitude = kwargs['magnitude']

    def __call__(self, name, arr):
        """Override () function to do Initialization
        Parameters
        name : str
            name of corrosponding ndarray
        arr : NDArray
            ndarray to be Initialized
        """
        if name.startswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            if self.method == 'xavier':
                mx.initializer.Xavier(self.rnd_type,self.factor_type,self.magnitude)._init_weight(name, arr)
        elif name.endswith('upper'):
            self._init_upper_bound(name, arr)
        elif name.endswith('lower'):
            self._init_lower_bound(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_one(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)
            
    def _init_upper_bound(self, name, arr):
        arr[:] = self.upper_bound

    def _init_lower_bound(self, name, arr):
        arr[:] = self.lower_bound
