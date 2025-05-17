import numpy as np

# Example training data
x_train = np.array([1, 2, 3, 4, 5])  # Features
y_train = np.array([3, 5, 7, 9, 11])  # Labels

# Initialize parameters
w = 0.0
b = 0.0

# Hyperparameters
alpha = 0.01  # learning rate
epochs = 1000  # number of iterations
m = x_train.shape[0]  # number of training examples

# Gradient Descent Loop
for epoch in range(epochs):
    # Compute predictions
    Y_pred = w * x_train + b
    
    # Compute gradients
    dw = (1/m) * np.sum((Y_pred - y_train) * x_train)
    db = (1/m) * np.sum(Y_pred - y_train)
    
    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

    cost = (1/(2*m)) * np.sum((Y_pred - y_train)**2)
    
    # Optionally, print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost = {cost:.7f}, w = {w:.4f}, b = {b:.4f}")

    if cost < 10e-6:  # Tolerance level for stopping criteria
        print(f"The tolerance has been met at epoch {epoch} with cost {cost:.4e}")
        break



print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

