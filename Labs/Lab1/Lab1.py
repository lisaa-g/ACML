import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate = 0.1, max_epochs = 100):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        # Initialise weights & bias randomly 
        self.weights = np.random.randn(num_inputs) # randn -> standard normal distribution (mean=0, variance=1)
        self.bias = np.random.randn()

    def predict(self, X):
        # input data X & predicts the output labels based on the learned weights & bias
        # dot product of input data with the weights, adds the bias, and applies a threshold function to predict the class labels (0 or 1
        return np.where(np.dot(X, self.weights) + self.bias > 0, 1, 0)

    def train(self, X, T): # trains the perceptron using given input X & target labels T
        N = len(X) # no. of rows
        for epoch in range(self.max_epochs):
            # Shuffle the indices
            I = np.random.permutation(N) # permutation() -> re-arranged array

            # iterate through shuffled indices & updates weights & bias based on perceptron learning rule
            for i in I:
                prediction = self.predict(X[i])
                if prediction != T[i]:
                    update = self.learning_rate * (T[i] - prediction)
                    self.weights += update * X[i]
                    self.bias += update

            # Compute loss
            loss = self.compute_loss(X, T)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {loss}")

    def compute_loss(self, X, T): # Computes misclassification loss of perceptron model
        predictions = self.predict(X)
        misclassified = np.sum(predictions != T) # compares predicted labels with target labels to count number of misclassifications
        return misclassified / len(T) # returns the ratio of misclassified data points to the total no. of data points as the loss

# Given dataset
X = np.array([[0, 0],
              [1, 1],
              [1, 0],
              [1, 1]])

T = np.array([1, 1, 1, 0])

# Learned weights and bias from the perceptron
perceptron = Perceptron(num_inputs=2)
perceptron.train(X, T)
weights = perceptron.weights
bias = perceptron.bias

# Decision boundary equation: w1*x1 + w2*x2 + b = 0
# Solving for x2: x2 = (-w1*x1 - b) / w2
x1 = np.linspace(-0.5, 1.5, 100)
x2 = (-weights[0] * x1 - bias) / weights[1]

# Plotting the data points
plt.scatter(X[:, 0], X[:, 1], c=T, cmap='bwr', label='Data Points')

# Plotting the decision boundary
plt.plot(x1, x2, color='black', label='Decision Boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linear Discriminant of Perceptron')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()