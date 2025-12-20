import numpy as np
from create_dataset import create_spiral_data

class ReLU:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Sigmoid:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.output = 1.0 / (1 + np.exp(-inputs,))
    
class Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def __init__(self):
        pass
    def calculate(self, outputs, y):
        self.losses = self.forward(outputs, y)
        data_loss = np.mean(self.losses)
        return data_loss

class Categorical_CrossEntropy(Loss):
    def __init__(self):
        pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # for categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = np.sum(y_pred_clipped[range(samples), y_true])

        # Mask values - for one hot labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    elif len(y_true.shape) == 1:
        y_true = y_true
    acc = np.mean(y_true == y_pred)
    print(acc)

def main():
    X, y = create_spiral_data(samples=300, classes=3)
    dense_1 = Layer_Dense(n_inputs=2, n_neurons=3)
    activation_1 = ReLU()
    activation_2 = Softmax()
    loss_fn = Categorical_CrossEntropy()

    dense_1.forward(X)
    activation_1.forward(dense_1.output)

    dense_2 = Layer_Dense(n_inputs=len(activation_1.output[0]), n_neurons=3)
    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.output)
    loss = loss_fn.calculate(activation_2.output, y)

    # print(activation_2.output[:5])
    # print(loss)
    accuracy(y, activation_2.output)

if __name__ == '__main__':

    main()