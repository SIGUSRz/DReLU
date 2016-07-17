#from Zoo.MLP2 import *
from Zoo.MLP import *
def build_model(name, setting):
    if name == 'mlp':
        return mlp(**setting)
    else:
        print 'Not Supported'
        return 0
