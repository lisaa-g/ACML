import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(input_data, weights_hidden, biases_hidden, weights_output, biases_output):
    # Compute activations of the hidden layer
    hidden_activations = np.dot(weights_hidden, input_data) + biases_hidden
    hidden_outputs = sigmoid(hidden_activations)
    
    # Compute activations of the output layer
    output_activations = np.dot(weights_output, hidden_outputs) + biases_output
    output_outputs = sigmoid(output_activations)
    
    return output_outputs

# Example weights and biases
input_size = 3
hidden_size = 4
output_size = 2

# Weight matrices
weights_hidden = np.random.randn(hidden_size, input_size)
weights_output = np.random.randn(output_size, hidden_size)

# Bias vectors
biases_hidden = np.random.randn(hidden_size, 1)
biases_output = np.random.randn(output_size, 1)

# Print weights and biases
print("Weights (Hidden Layer):")
print(weights_hidden)
print("\nBiases (Hidden Layer):")
print(biases_hidden)
print("\nWeights (Output Layer):")
print(weights_output)
print("\nBiases (Output Layer):")
print(biases_output)

# Input data
input_data = np.random.randn(input_size, 1)
print(f'\nInput: {input_data}')

# Feedforward step
output = feedforward(input_data, weights_hidden, biases_hidden, weights_output, biases_output)
print("\nOutput:", output)
