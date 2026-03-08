import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for step in range(steps):
        # Compute model
        model = 1/(1 + np.exp(-(X @ w + b)))
        
        # Compute gradients
        dw = (1/n_samples) * (X.T @ (model - y))
        db = (1/n_samples) * np.sum(model - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b
