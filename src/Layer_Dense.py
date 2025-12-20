import numpy as np
from create_dataset import create_spiral_data
from Layer_ReLU import ReLU
from Layer_Softmax import Softmax
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.01 * np.random.randn(self.n_inputs, self.n_neurons)
        # print(self.weights.shape)
        self.biases = np.zeros((1, self.n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # print(self.output.shape)
    

if __name__ == '__main__':
    X, y = create_spiral_data(samples=300, classes=3)
    dense_1 = Layer_Dense(n_inputs=2, n_neurons=3)
    
    activation_relu = ReLU()
    activation_softmax = Softmax()

    dense_1.forward(X)
    activation_relu.forward(dense_1.output)
    dense_2 = Layer_Dense(n_inputs=len(dense_1.output[0]), n_neurons=3)
    dense_2.forward(activation_relu.output)
    activation_softmax.forward(dense_2.output)
    print(activation_softmax.output[:5])