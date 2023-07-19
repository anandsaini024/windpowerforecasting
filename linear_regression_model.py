import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_value=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.lambda_value = lambda_value

    def fit(self, X, y):
        X= np.array(X)
        num_samples, num_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            # Predict using current weights and bias
            y_pred = self.predict(X)

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            if self.regularization == 'l1':
                dw += (self.lambda_value / num_samples) * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += (self.lambda_value / num_samples) * self.weights
        
            # Check for NaN or infinite values
            if np.isnan(dw).any() or np.isnan(db).any() or np.isinf(dw).any() or np.isinf(db).any():
                print("NaN or infinite values encountered during gradient calculation. Skipping iteration.")
                continue
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Predict the target variable
        return np.dot(X, self.weights) + self.bias

