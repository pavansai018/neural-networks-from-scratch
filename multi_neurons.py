
def multi_neurons(inputs, weights, biases):
    """
    Simulates multiple neurons in a neural network layer.

    Parameters:
    inputs (list of float): The input values to the neurons.
    weights (list of list of float): The weights associated with each input for each neuron.
    biases (list of float): The bias terms for each neuron.

    Returns:
    list of float: The outputs of the neurons after applying the activation function.
    """
    outputs = []
    for neuron_weights, bias in zip(weights, biases):
        # Calculate the weighted sum of inputs plus bias for each neuron
        weighted_sum = sum(i * w for i, w in zip(inputs, neuron_weights)) + bias
        
        # Apply the activation function (ReLU in this case)
        output = max(0, weighted_sum)
        outputs.append(output)
    
    return outputs

# Example usage
if __name__ == "__main__":
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, 0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    biases = [2.0, 3.0, 0.5]
    results = multi_neurons(inputs, weights, biases)
    print(f"Neurons outputs: {results}")    