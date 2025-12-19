import numpy as np

def single_neuron(inputs, weights, bias):
    '''
    Docstring for single_neuron
    Neuron dot product with numpy
    '''
    weighted_sum = np.dot(inputs, weights) + bias
    return weighted_sum

if __name__ == '__main__':
    i = [1, 2, 3]
    w = [0.2, 0.8, -0.2]
    b = 2.0
    output = single_neuron(inputs=i, weights=w, bias=b)
    print(f'Output of Dot Product: {output}')