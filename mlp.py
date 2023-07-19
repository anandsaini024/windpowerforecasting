import numpy as np


def train_mlp(X, y, hidden_sizes, learning_rate, num_epochs):
    # Initialize weights
    layer_sizes = [X.shape[1]] + hidden_sizes + [1]
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        # Xavier initialization
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1]))
        b = np.zeros((1, layer_sizes[i+1]))
        weights.append(W)
        biases.append(b)

    # Training loop
    for epoch in range(num_epochs):
        # Forward propagation
        activations = forward_propagation(X, weights, biases)

        # Backpropagation
        gradients = backward_propagation(X, y, activations, weights, biases)
        weights, biases = update_weights(weights, biases, gradients, learning_rate)

        # Print training loss
        loss = mean_squared_error(y, activations[-1])
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

    return weights, biases


def initialize_weights(input_size, hidden_sizes, output_size):
    sizes = [input_size] + hidden_sizes + [output_size]
    weights = []
    biases = []
    for i in range(len(sizes) - 1):
        W = np.random.randn(sizes[i], sizes[i+1])
        b = np.zeros((1, sizes[i+1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

def forward_propagation(X, weights, biases):
    activations = [X]
    for i in range(len(weights)):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        A = relu(Z)
        activations.append(A)
    return activations

def relu(Z):
    return np.maximum(0, Z)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def backward_propagation(X, y, activations, weights, biases):
    gradients = []
    m = X.shape[0]
    dZ = activations[-1] - y

    for i in range(len(weights)-1, -1, -1):
        dW = np.dot(activations[i].T, dZ) / m
        db = np.mean(dZ, axis=0)
        gradients.append((dW, db))
        
        if i > 0:
            dZ = np.dot(dZ, weights[i].T) * (activations[i] > 0)

    gradients.reverse()
    return gradients

def update_weights(weights, biases, gradients, learning_rate):
    updated_weights = []
    updated_biases = []
    for i in range(len(weights)):
        W = weights[i] - learning_rate * gradients[i][0]
        b = biases[i] - learning_rate * gradients[i][1]
        updated_weights.append(W)
        updated_biases.append(b)
    return updated_weights, updated_biases
def predict_mlp(X, weights, biases):
    activations = forward_propagation(X, weights, biases)
    return activations[-1]