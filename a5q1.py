import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt("ex2data1.txt", delimiter=',')
X = data[:, 0]  # Population
y = data[:, 1]  # Profit
m = len(y)
print("Number of training examples:", m)
print(f"Input features: {X}")
print(f"Output labels: {y}")

# Add intercept term to X
X_b = np.c_[np.ones((m, 1)), X]
print("X with intercept term:\n", X_b)
theta = np.zeros(2)
alpha = 0.01
iterations = 1500

def compute_cost(X, y, theta):
    errors = X @ theta - y
    return (1 / (2 * m)) * np.dot(errors, errors)

def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= alpha * gradient
    return theta

# Train the model
theta = gradient_descent(X_b, y, theta, alpha, iterations)
print("Learned theta:", theta)


# Optional: plot the result
plt.scatter(X, y, color='red', marker='x', label='Training data')
plt.plot(X, X_b @ theta, label='Linear regression')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.legend()


# Grid over which we will calculate J
theta0_vals = np.linspace(-30, 30, 100)
theta1_vals = np.linspace(-10, 10, 100)

# Initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(X_b, y, t)

# Transpose J_vals before plotting since matplotlib's contour expects (X, Y)
J_vals = J_vals.T

# Contour plot
plt.figure(figsize=(8, 6))
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2, label='Minimum')
plt.title('Contour plot of cost function')
plt.legend()
plt.grid(True)


from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid for theta0 and theta1
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

# J_vals is already computed above, no need to recompute
# Remember: J_vals was transposed for the contour plot

# Surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('Cost J')
ax.set_title('Surface plot of cost function')
plt.show()




plt.show()
