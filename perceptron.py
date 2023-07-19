import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.001):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_inputs)
        self.bias = 0

    def predict(self, X):
        # Calculate the weighted sum of inputs and add bias
        z = np.dot(X, self.weights) + self.bias
        return z

    def train(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for i in range(X.shape[0]):
                # Compute the predicted value
                y_pred = self.predict(X[i])

                # Compute the error
                error = y[i] - y_pred

                # Update the weights and bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
