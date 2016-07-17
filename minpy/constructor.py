from model_zoo.MLP2 import *
def build_model(name, setting):
    if name == 'mlp':
        return mlp(**setting)
    else:
        print 'Not Supported'
        return 0
