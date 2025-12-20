'''
Docstring for multi_layer_neurons
4 input features
2 hidden layers with 3 neurons in each layer
'''
import numpy as np
def multi_layer(inputs, weights1, biases1, weights2, biases2):
    layer_1_output = np.dot(inputs, np.array(weights1).T) + biases1
    layer_2_output = np.dot(layer_1_output, np.array(weights2).T) + biases2
    return layer_2_output

if __name__ == '__main__':
    i = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]

    w1 = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    b1 = [2.0, 3.0, 0.5]
    w2 = [
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33,],
        [-0.44, 0.73, -0.13,]
    ]
    b2 = [-1.0, 2.0, -0.5]
    print(f'Weighted Sum: {multi_layer(i, w1, b1, w2, b2)}')
