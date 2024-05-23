import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
            # Shuffle the dataset
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs = inputs[indices]
            targets = targets[indices]
            
            total_error = 0
            
            for x, t in zip(inputs, targets):
                x = np.array(x)
                t = np.array(t)
                self.feedforward(x)
                self.backpropagate(x, t)
                
                total_error += np.sum((self.output_activation - t) ** 2)
                
            mean_error = total_error / len(inputs)
            print(f"Epoch {epoch+1}/{epochs}, Mean Squared Error: {mean_error:.6f}")
            

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode target labels
encoder = OneHotEncoder(categories='auto')
y_encoded = encoder.fit_transform(y).toarray()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize input features
X_train /= np.max(X_train, axis=0)
X_test /= np.max(X_test, axis=0)

# Create neural network
input_size = X_train.shape[1]
hidden_size = 8
output_size = y_encoded.shape[1]
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the network
nn.train(X_train, y_train, epochs=100)

# Test the network
total_error = 0
for x, t in zip(X_test, y_test):
    x = np.array(x)
    t = np.array(t)
    output = nn.feedforward(x)
    total_error += np.sum((output - t) ** 2)
    
mean_error = total_error / len(X_test)
print(f"Mean Squared Error on Test Dataset: {mean_error:.6f}")
