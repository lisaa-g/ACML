import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases with random values
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)
        
        # Learning rate
        self.learning_rate = 0.1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, x):
        # Input to hidden layer
        self.hidden_activation = self.sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        
        # Hidden to output layer
        self.output_activation = self.sigmoid(np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output)
        
        return self.output_activation
    
    def backpropagate(self, x, target):
        # Compute output layer delta
        output_delta = (self.output_activation - target) * self.sigmoid_derivative(self.output_activation)
        
        # Compute hidden layer delta
        hidden_delta = np.dot(output_delta, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_activation)
        
        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * np.outer(self.hidden_activation, output_delta)
        self.weights_input_hidden -= self.learning_rate * np.outer(x, hidden_delta)
        
        self.bias_output -= self.learning_rate * output_delta
        self.bias_hidden -= self.learning_rate * hidden_delta
        
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
hidden_size = 3
output_size = 1

# Create dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Create neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the network
nn.train(inputs, targets, epochs=1000)

# Test the network
for x in inputs:
    print(f"Input: {x}, Predicted Output: {nn.feedforward(x)}")
