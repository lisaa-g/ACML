import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(input_values, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Convert input_values to a numpy array for matrix multiplication
    input_vector = np.array(input_values)

    # Compute output of the hidden layer
    hidden_layer_output = sigmoid(np.dot(input_vector, weights_input_hidden) + bias_hidden)

    # Compute final output
    output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output) + bias_output)

    return output

# Specify your own weights and biases
weights_input_hidden = np.array([[2, 1, -1], [-2, -4, 1]])
bias_hidden = np.array([1, -1, 2])
weights_hidden_output = np.array([[-2, 3], [3, -1], [5, 0]])
bias_output = np.array([0, -1])

# Example input values
input_values = [2, 1]

# Compute the output of the neural network
output_result = feedforward(input_values, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

print("Input Values:", input_values)
print("Output Result:", output_result)
