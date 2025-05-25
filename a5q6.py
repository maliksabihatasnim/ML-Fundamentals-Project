import pandas as pd


digits_data = pd.read_csv("digits_data.csv")

# Display basic info and first few rows
digits_data.info(), digits_data.head()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Split into features and labels
X = digits_data.drop(columns=["label"]).values
y = digits_data["label"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: One-vs-All logistic regression
# Train one classifier per class
classifiers = []
num_classes = len(np.unique(y_train))
for c in range(num_classes):
    y_binary = (y_train == c).astype(int)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_binary)
    classifiers.append(clf)

# Step 3: Predict using all classifiers and choose the class with highest score
# Predict probabilities for all classes
prob_matrix = np.array([clf.predict_proba(X_train)[:, 1] for clf in classifiers]).T
y_train_pred = np.argmax(prob_matrix, axis=1)

# Compute training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)
print(training_accuracy)


import matplotlib.pyplot as plt

# Step 4: Learned parameters visualization
# We'll visualize the weights of each classifier as 8x8 images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Learned Parameters for Each Class (One-vs-All)", fontsize=14)

for i, clf in enumerate(classifiers):
    weights = clf.coef_.reshape(8, 8)
    ax = axes[i // 5, i % 5]
    cax = ax.imshow(weights, cmap='seismic', interpolation='nearest')
    ax.set_title(f'Class {i}')
    ax.axis('off')

fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)


# Visualize some sample digits from the training set
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Sample Digits from Training Set", fontsize=14)

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_train[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()


# Use scikit-learn's built-in LogisticRegression with multi_class='ovr' for comparison
clf_builtin = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
clf_builtin.fit(X_train, y_train)

# Predict on training data
y_train_pred_builtin = clf_builtin.predict(X_train)

# Compute training accuracy
training_accuracy_builtin = accuracy_score(y_train, y_train_pred_builtin)
training_accuracy_builtin
