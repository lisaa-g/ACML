import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Use provided weights and biases, rounded to 3 decimal places
        self.weights_input_hidden = np.round(weights_input_hidden.astype(float), 3)
        self.weights_hidden_output = np.round(weights_hidden_output.astype(float), 3)
        self.bias_hidden = np.round(bias_hidden.astype(float), 3)
        self.bias_output = np.round(bias_output.astype(float), 3)
        
        # Learning rate
        self.learning_rate = 0.1
        
    def sigmoid(self, x):
        return np.round(1 / (1 + np.exp(-x)), 3)
    
    def sigmoid_derivative(self, x):
        return np.round(x * (1 - x), 3)
    
    def feedforward(self, x):
        # Input to hidden layer
        self.hidden_activation = self.sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        
        # Hidden to output layer
        self.output_activation = self.sigmoid(np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output)
        
        return np.round(self.output_activation, 3)
    
    def backpropagate(self, x, target):
        # Compute output layer delta
        output_delta = np.round((self.output_activation - target) * self.sigmoid_derivative(self.output_activation), 3)
        
        # Compute hidden layer delta
        hidden_delta = np.round(np.dot(output_delta, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_activation), 3)
        
        # Update weights and biases
        self.weights_hidden_output = np.round(self.weights_hidden_output - self.learning_rate * np.outer(self.hidden_activation, output_delta), 3)
        self.weights_input_hidden = np.round(self.weights_input_hidden - self.learning_rate * np.outer(x, hidden_delta), 3)
        
        self.bias_output = np.round(self.bias_output - self.learning_rate * output_delta, 3)
        self.bias_hidden = np.round(self.bias_hidden - self.learning_rate * hidden_delta, 3)
        
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for x, t in zip(inputs, targets):
                x = np.array(x)
                t = np.array(t)
                self.feedforward(x)
                self.backpropagate(x, t)
            print(f"Epoch {epoch+1}/{epochs} completed")
            

# Example usage
input_size = 2
hidden_size = 2
output_size = 2

# Provided weights and biases
weights_input_hidden = np.array([[1, -1],
                                  [-3, -2]])

weights_hidden_output = np.array([[-2, 0],
                                   [0, 3]])

bias_hidden = np.array([[0, 1]])
bias_output = np.array([[1, -2]])

# Create dataset
inputs = np.array([[1, -2]])
targets = np.array([[1, 0]])

# Create neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

# Train the network
nn.train(inputs, targets, epochs=1)

# Test the network
for x in inputs:
    print(f"Input: {x}, Predicted Output: {nn.feedforward(x)}")
