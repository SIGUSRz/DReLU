import mxnet as mx
import numpy as np
bound_record = {}

class drelu(mx.operator.NumpyOp):
    def __init__(self,name):
        #inheriting from superclass NumpyOp
        #Override operator functions defined in mxnet operator.py CustomOp class
        super(drelu,self).__init__(True)
        self.name = name
        print name
    
    def list_arguments(self):
        return ['data', 'lower', 'upper']

    def list_outputs(self):
        return ['output']

    def infer_shape(self,input_shape):
        #Return input_shape and output_shape
        return [input_shape[0], (1,), (1,)], [input_shape[0]]

    def forward(self, **kwargs):
        inputs = kwargs['in_data']
        outputs = kwargs['out_data']
        lower_bound, upper_bound = inputs[1][0],inputs[2][0]
        outputs[0][:] = np.minimum(upper_bound, np.maximum(lower_bound, inputs[0]))

    def backward(self, **kwargs):
        #inputs: [0] data [1] lower_bound [2] upper_bound
        inputs = kwargs['in_data']
        #outputs: [0] activation value
        outputs = kwargs['out_data']
        #[0] dx [1] dlower [2] dupper
        input_gradient = kwargs['in_grad']
        #[0] dactivation
        output_gradient = kwargs['out_grad']

        lower_bound, upper_bound = inputs[1][0],inputs[2][0]
        activations = outputs[0]
        
        dx = input_gradient[0]
        dx[:] = output_gradient[0]
        dx[activations==lower_bound] = 0.0
        dx[activations==upper_bound] = 0.0

        dout = output_gradient[0]
        dlower_bound = input_gradient[1]
        dupper_bound = input_gradient[2]
        dlower_bound[:] = np.sum(dout[activations==lower_bound])
        dupper_bound[:] = np.sum(dout[activations==upper_bound])
