import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 5, 7, 9, 11])
m = X.shape[0]

# Define the cost function
def compute_cost(w, b, X, Y):
    Y_pred = w * X + b
    cost = (1/(2*m)) * np.sum((Y_pred - Y)**2)
    return cost

# Create a grid of w and b values
w_vals = np.linspace(-1, 5, 100)
b_vals = np.linspace(-5, 5, 100)
W, B = np.meshgrid(w_vals, b_vals)

J_vals = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J_vals[i, j] = compute_cost(W[i, j], B[i, j], X, Y)
print(J_vals.shape)
# Perform gradient descent and store (w, b) history
w = 0.0
b = 0.0
alpha = 0.01
epochs = 1000

w_history = []
b_history = []

for epoch in range(epochs):
    Y_pred = w * X + b
    dw = (1/m) * np.sum((Y_pred - Y) * X)
    db = (1/m) * np.sum(Y_pred - Y)
    
    w = w - alpha * dw
    b = b - alpha * db
    
    w_history.append(w)
    b_history.append(b)

# Plot contour
plt.figure(figsize=(10, 6))
plt.contour(W, B, J_vals, levels=np.logspace(-1, 3, 20), cmap="jet")
plt.xlabel('w')
plt.ylabel('b')
plt.title('Contour plot of Cost(w, b)')

# Plot (w,b) history
plt.scatter(w_history, b_history, color='red', s=10, label='Gradient Descent Path')
plt.legend()
plt.show()
