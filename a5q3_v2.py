import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#                                           (a) Visualize data using a scatter plot
# Load data from file
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]  # features
y = data[:, 2]    # target/output
m = len(y)

# Scaling
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add intercept term (bias)
X_b = np.hstack([np.ones((m, 1)), X])

fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y,cmap = 'bwr', edgecolor="black")
ax.set(title="Student Admission Based on Exam Scores", xlabel="Exam Score 1", ylabel="Exam Score 2")
legend = ax.legend(*scatter.legend_elements(), title="Admitted")
ax.add_artist(legend)


#                                                  (b) Hypothesis: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#                           (c) Cost function for logistic regression (log loss or binary cross-entropy)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost


#                                   (d)Implement Gradient Descent function & Find Optimal Parameters
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


theta_init = np.zeros((X_b.shape[1])) # column no of X_b
alpha = 0.01
iterations = 5000
theta_optimal, cost_history = gradient_descent(X_b, y, theta_init, alpha, iterations)
print(f"Learned parameter: theta = {theta_optimal}")

#Visualize h(θT*x)= 1/(1+exp(-θT*x)) vs θT*x
z = X_b @ theta_optimal
h = sigmoid(z)
plt.figure(figsize=(8, 6))
plt.plot(z,h,'b.')

plt.ylim(0, 1.1)
plt.xlabel(r'$\theta^T x$')
plt.ylabel(r'$h(\theta^T x)$')
plt.title(r'Sigmoid Function: $h(\theta^T x) = \frac{1}{1 + \exp(-\theta^T x)}$')
plt.grid(False)
plt.axhline(0.5, color='k', linestyle='--',linewidth=2,label = "Threshold Probability h=0.5")
plt.axvline(0, color='teal', linestyle='--',linewidth=2, label = "Decision Boundary θT*x=0")
plt.legend(loc='best')



#                                           (e) Plot the decision boundary)
b,w1,w2 = theta_optimal
# The decision boundary is defined by the equation: 0 = w1*x1 + w2*x2 + b
# Rearranging gives us: x2 = -w1/w2*x1 - b/w2
c = -b/w2
m = -w1/w2

x1min = X[:,0].min()
x1max = X[:,0].max()
x2min = X[:,1].min()
x2max = X[:,1].max()
X1 = np.array([x1min-0.5, x1max+0.5])
X2 = m*X1 + c

fig, ax = plt.subplots(figsize=(6, 4))
plt.plot(X1,X2,'k--',linewidth=2)
plt.fill_between(X1, X2, x2min-0.5, color='tab:blue', alpha=0.4)
plt.fill_between(X1, X2, x2max+0.5, color='tab:red', alpha=0.4)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y,cmap = 'bwr', edgecolor="black")


ax.set_xlim(x1min-0.5,x1max+0.5)
ax.set_ylim(x2min-0.5,x2max+0.5)
ax.set(title='Decision Boundary: Logistic Regression', xlabel="Exam Score 1", ylabel="Exam Score 2")
legend = ax.legend(*scatter.legend_elements(), title="Admitted")
ax.add_artist(legend)
plt.grid(False)


#                                       (f) Evaluate accuracy on the training set
def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    return probabilities >= 0.5

predictions = predict(X_b, theta_optimal)
accuracy = (predictions == y).mean() * 100
print(f"The accuracy on the training set: {accuracy} %")

# Using Built-in Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

logistic_regression_multinomial = LogisticRegression().fit(X, y)
logistic_regression_ovr = OneVsRestClassifier(LogisticRegression()).fit(X, y)

accuracy_multinomial = logistic_regression_multinomial.score(X, y)
accuracy_ovr = logistic_regression_ovr.score(X, y)

from sklearn.inspection import DecisionBoundaryDisplay

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for model, title, ax in [
    (
        logistic_regression_multinomial,
        f"Multinomial Logistic Regression\n(Accuracy: {accuracy_multinomial:.3f})",
        ax1,
    ),
    (
        logistic_regression_ovr,
        f"One-vs-Rest Logistic Regression\n(Accuracy: {accuracy_ovr:.3f})",
        ax2,
    ),
]:
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        ax=ax,
        response_method="predict",
        alpha=0.6,
        cmap="bwr",
        xlabel="Exam Score 1",
        ylabel="Exam Score 2",
    )
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y,cmap = 'bwr', edgecolor="black")
    legend = ax.legend(*scatter.legend_elements(), title="Admitted")
    ax.add_artist(legend)
    ax.set_title(title)
  

plt.show()