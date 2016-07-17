import minpy.numpy as np

def gaussian_random(structure,mode,weight,bias,lower,upper):
    param = []
    drelu_pos = []

    for layer in range(len(structure) - 1):
        param.append(np.random.normal(0, weight, (structure[layer],structure[layer+1])))
        param.append(np.full(structure[layer + 1], bias))
    for layer in range(len(structure) - 2):
        if mode[layer] == 'drelu':
            param.append(fixed_bound(structure[layer + 1], lower))
            param.append(fixed_bound(structure[layer + 1], upper))
            drelu_pos.append(layer)
    return param, drelu_pos

def fixed_bound(structure, value):
    return np.full(structure, value)
