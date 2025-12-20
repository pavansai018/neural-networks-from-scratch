import numpy as np

def layer_of_neurons(inputs, weights, biases):
    return np.dot(inputs, np.array(weights).T) + biases

if __name__ == '__main__':
    i = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]

    w = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    b = [2.0, 3.0, 0.5]
    print(f'Weighted Sum: {layer_of_neurons(i, w, b)}')