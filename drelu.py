import mxnet as mx
import numpy as np
bound_record = {}

class drelu(mx.operator.NumpyOp):
    def __init__(self, name, implement)
