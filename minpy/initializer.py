import minpy.numpy as np

def gaussian_random(hidden_size, weight_range=(0, 0.05), bias_value=0.0):
    param = []
    for layer in range(len(hidden_size)-1):
        param.append(np.random.normal(weight_range[0],weight_range[1],(hidden_size[layer],hidden_size[layer+1])))
        param.append(np.full(hidden_size[layer+1], bias_value))
    return param
