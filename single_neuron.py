
def single_neuron(inputs, weights, bias):
    """
    Simulates a single neuron in a neural network.

    Parameters:
    inputs (list of float): The input values to the neuron.
    weights (list of float): The weights associated with each input.
    bias (float): The bias term for the neuron.

    Returns:
    float: The output of the neuron after applying the activation function.
    """
    # Calculate the weighted sum of inputs plus bias
    weighted_sum = sum(i * w for i, w in zip(inputs, weights)) + bias
    
    # Apply the activation function (ReLU in this case)
    output = max(0, weighted_sum)
    
    return output

# Example usage
if __name__ == "__main__":
    inputs = [1.0, 2.0, 3.0]
    weights = [0.2, 0.8, -0.5]
    bias = 2.0
    result = single_neuron(inputs, weights, bias)
    print(f"Neuron output: {result}")