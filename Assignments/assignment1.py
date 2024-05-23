import numpy as np

def initialize_parameters():
    #initialize weights and biases
    weights_input_hidden = np.ones((4, 8))
    biases_hidden = np.ones(8)
    weights_hidden_output = np.ones((8, 3))
    biases_output = np.ones(3)
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def loss(predicted, target):
    return 0.5 * np.sum((predicted - target) ** 2)

def feedforward_hidden(inputs, weights_input_hidden, biases_hidden):
    #input layer to hidden layer
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid_activation(hidden_layer_input)
    return hidden_layer_output

def feedforward_output(hidden_layer_output, weights_hidden_output, biases_output):
    #hidden layer to output layer
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    output_layer_output = sigmoid_activation(output_layer_input)
    return output_layer_output

def backpropagation(inputs, target, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate=0.1):
    hidden_layer_output = feedforward_hidden(inputs, weights_input_hidden, biases_hidden)
    output_layer_output = feedforward_output(hidden_layer_output, weights_hidden_output, biases_output)
    #output layer error
    output_errors = (output_layer_output - target) * output_layer_output * (1 - output_layer_output)
    #backpropagation error
    hidden_errors = np.dot(output_errors, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)
    #update weights and biases
    weights_hidden_output = weights_hidden_output - learning_rate * np.dot(hidden_layer_output.T, output_errors)
    biases_output = biases_output - learning_rate * output_errors.sum(axis=0)
    weights_input_hidden = weights_input_hidden- learning_rate * np.dot(inputs.T, hidden_errors)
    biases_hidden = biases_hidden - learning_rate * hidden_errors.sum(axis=0)

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

if __name__ == "__main__":
    #input and target
    data = [float(input()) for _ in range(7)]
    inputs = np.array([data[:4]])
    target = np.array([data[4:]])

    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = initialize_parameters()
    hidden_layer_output = feedforward_hidden(inputs, weights_input_hidden, biases_hidden)
    initial_output = feedforward_output(hidden_layer_output, weights_hidden_output, biases_output)
    initial_loss = loss(initial_output, target)
    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = backpropagation(inputs, target, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)
    #updated loss
    hidden_layer_output = feedforward_hidden(inputs, weights_input_hidden, biases_hidden)
    updated_output = feedforward_output(hidden_layer_output, weights_hidden_output, biases_output)
    updated_loss = loss(updated_output, target)

    print(round(initial_loss, 4))
    print(round(updated_loss, 4))