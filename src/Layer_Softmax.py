import numpy as np

class Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        # print(exp_values)
        # print(np.sum(exp_values, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


if __name__ == '__main__':
    i = [
        [1, 2, 3, 4, 5],
        [0.564, 4.56, 3.23, 4.23, 0.987]
    ]
    a = Softmax()
    a.forward(i)
    print(a.output)