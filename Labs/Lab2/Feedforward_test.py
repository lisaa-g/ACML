import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(input_data, weights_hidden, biases_hidden, weights_output, biases_output):
    # Compute activation values of the hidden layer
    hidden_activations = np.dot(input_data, weights_hidden) + biases_hidden
    hidden_outputs = sigmoid(hidden_activations)
    
    # Compute activation values of the output layer
    output_activations = np.dot(hidden_outputs, weights_output) + biases_output
    output_outputs = sigmoid(output_activations)
    
    return output_outputs

# Example weights and biases
input_size = 4
hidden_size = 4
output_size = 2

# Weight matrices
weights_hidden = np.array([[4,-5,0,1],
                  [-3,6,-1,2],
                  [0,1,1,-2],
                  [2,0,-3,4]])

weights_output = np.array([[-2,1],
                  [-1,0],
                  [1,-3],
                  [5,-1]])

# Bias vectors
biases_hidden = np.array([[1,-2,0,-1]])
biases_output = np.array([[-2,2]])

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
input_data = np.array([[-2,3,1,0]])
print(f'\nInput: {input_data}')

# Feedforward step
output = feedforward(input_data, weights_hidden, biases_hidden, weights_output, biases_output)
print(f"\nOutput: {np.round(output,3)}")
