import numpy as np

def multi_neuron(inputs, weights, biases):
    '''
    Docstring for multi_neuron
    
    :param inputs: inputs
    :param weights: weights
    :param bias: biases

    Returns:
    -------------
    weighted_sum
    '''
    return np.dot(weights, inputs) + biases

if __name__ == '__main__':
    i = [1, 2, 3, 2.5]
    w = [   [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26,-0.5],
            [-0.26, -0.27, 0.17,0.87]]
    b = [2, 3, 0.5]
    print(f'Weighted Sum: {multi_neuron(i, w, b)}')