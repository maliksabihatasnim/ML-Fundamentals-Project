import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data from file
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]  # features
y = data[:, 2]    # target/output
m = len(y)

# Scaling
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

#                       (a) Map features to polynomial features including intercept term
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_b = poly.fit_transform(X)

# Visualization
fig, ax = plt.subplots(figsize=(6, 4))

scatter = ax.scatter(X[:, 0], X[:, 1], c=y,cmap = 'bwr', edgecolor="black")
ax.set(title="Microchip Quality Test Result", xlabel="Test Score 1", ylabel="Test Score 2")
legend = ax.legend(*scatter.legend_elements(), title="Fail/Pass")
ax.add_artist(legend)

# Hypothesis: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function for logistic regression (log loss or binary cross-entropy)
def reg_compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost

# Gradient Descent function
def reg_gradient_descent(X, y, theta, alpha,lmd,iterations):
    m = len(y)
    cost_history = []
    theta = np.array(theta)

    for i in range(iterations):
        h = sigmoid(X @ theta)
        grad = (1/m) * (X.T @ (h - y))

        # Regularization term for gradient
        theta[0] -= alpha * grad[0]
        theta[1:] -= alpha * (grad[1:] - (lmd/m) * theta[1:])
        cost = reg_compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta,cost_history


# Run gradient descent
alpha = 0.01
iterations = 5000
lmd = 1
theta_init = np.zeros((X_b.shape[1])) # column no of X_b
theta_optimal,cost_history = reg_gradient_descent(X_b, y, theta_init, alpha,lmd,iterations)

# Visualize h(θT*x)= 1/(1+exp(-θT*x)) vs θT*x
z = X_b @ theta_optimal
h = sigmoid(z)
plt.figure(figsize=(8, 6))
plt.plot(z,h,'b.')
plt.axhline(0.5, color='k', linestyle='--',linewidth=2,label = "Threshold Probability h=0.5")
plt.axvline(0, color='teal', linestyle='--',linewidth=2, label = "Decision Boundary θT*x=0")
plt.legend(loc='best')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('h(z)')
plt.title('Sigmoid Function')
plt.grid(False)




#                             (d) Plot the decision boundary for polynomial features.
x1min = X[:,0].min()
x1max = X[:,0].max()
x2min = X[:,1].min()
x2max = X[:,1].max()


u = np.linspace(x1min - 0.5, x1max + 0.5, 100)
v = np.linspace(x2min - 0.5, x2max + 0.5, 100)
z_plot = np.zeros((len(u), len(v)))


for i in range(len(u)):
    for j in range(len(v)):
        # Create a single data point with the two features
        point = np.array([[u[i], v[j]]])
        # Apply the same polynomial transformation
        point_poly = poly.transform(point)
        # Calculate the linear combination X_b @ theta
        z_plot[i, j] = (point_poly @ theta_optimal)[0]


fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y,cmap = 'bwr', edgecolor="black")

# Plot the decision boundary (contour line where z = 0)
ax.contour(u, v, z_plot.T, levels=[0], colors='k', linestyles='--', linewidths=2)
ax.contourf(u, v, z_plot.T, levels=[-np.inf, 0, np.inf], colors=['tab:blue', 'tab:red'], alpha=0.4)


ax.set_xlim(x1min-0.5,x1max+0.5)
ax.set_ylim(x2min-0.5,x2max+0.5)
ax.set(title='Decision Boundary: Regularized Logistic Regression', xlabel="Test Score 1", ylabel="Test Score 2")
legend = ax.legend(*scatter.legend_elements(), title="Fail/Pass")
ax.add_artist(legend)
plt.grid(False)


# Learned Theta & Accuracy for Different values of regularization parameter λ
def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    return probabilities >= 0.5


lmd = [0,1,10]
for i in range(len(lmd)):
    theta_init = np.zeros((X_b.shape[1])) # column no of X_b
    theta_optimal,cost_history = reg_gradient_descent(X_b, y, theta_init, alpha,lmd[i],iterations)
    print(f"Learned parameter for lambda = {lmd[i]}: theta = {theta_optimal}")
    predictions = predict(X_b, theta_optimal)
    accuracy = (predictions == y).mean() * 100
    print(f"The accuracy on the training set: {accuracy} %")


#                                       (e) Different λ values effects
print("\n--- Effect of λ on Model ---\n")
print("λ = 0 (No Regularization)")
print("The decision boundary is very complex and tightly follows the training data.")
print("This results in overfitting, where the model performs well on training data but may generalize poorly to new/unseen data.\n")

print("λ = 1 (Moderate Regularization)")
print("The decision boundary is smooth and still separates the classes reasonably well.")
print("This is likely a good balance between bias and variance, i.e., it prevents overfitting while maintaining decent accuracy,often the best choice.\n")

print("λ = 100 (High Regularization)")
print("The boundary is almost linear or under-fitted to the data.")
print("It fails to capture the nonlinear decision boundary, leading to underfitting and high bias.")

plt.show()
