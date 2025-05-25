import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

m = 100
np.random.seed(0)
a = 0
b = 10
X = np.sort(np.random.rand(m, 1) * (b - a) + a, axis=0)  
y = X * np.sin(X) + np.random.randn(m, 1) * 2


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_val = scaler.transform(X_val)



def create_model(deg, alpha):
    model = make_pipeline(
        
        PolynomialFeatures(degree=degree),
        
        Ridge(alpha=alpha,solver='lsqr')
    )
    ridge_model = model.named_steps['ridge']
    return model, ridge_model

def plot_learning_curves(deg, alpha, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []
    model, ridge_model= create_model(deg, alpha)
    for m in range(1, len(X_train)):  
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    return train_errors, val_errors

def plot_fitting(deg,alpha):
    model,ridge_model = create_model(deg,alpha)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_val)

    X_fit = np.linspace(a,b,m).reshape(-1,1)
    X_fit = scaler.transform(X_fit)
    y_fit = model.predict(X_fit)
    
    return X_fit, y_fit, y_pred_train, y_pred_test



alpha = 1
degree = (2,10)
model,ridge_model = create_model(degree,alpha)
model.fit(X_train, y_train)

print("")
intercept = ridge_model.intercept_
coefficients = ridge_model.coef_
print(f"For lambda = {alpha} & degree between {degree[0]} and {degree[1]}:")
print("Ridge Regression Model Coefficients:")
print("Learned Intercept:", intercept)
print("Learned Parameters:", coefficients)
print("")


import matplotlib.pyplot as plt
import numpy as np
lmd = [0.1,1,10,100]
ind = 0

fig, axes = plt.subplots(2,2, figsize=(14, 10)) 
fig1,ax1 = plt.subplots(1,1, figsize=(14, 10))
ax1.scatter(X_train, y_train, label='Training Data', alpha=0.5)
ax1.scatter(X_val, y_val, label='Validation Data', alpha=0.5)

for i in range(2):
    for j in range(2):
        ax = axes[i, j]  
        alpha = lmd[ind]
        ind = ind + 1
        
        train_errors, val_errors = plot_learning_curves(degree, alpha, X_train, y_train, X_val, y_val)
        print(f"Training errors for lambda = {alpha}:",train_errors[:5])
        print(f"Validation errors for lambda = {alpha}:",val_errors[:5])
        
        ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training error")
        ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation error")
        ax.legend(loc="best", fontsize=14)
        ax.set_xlabel("Training set size", fontsize=14)
        ax.set_ylabel("RMSE", fontsize=14)
        ax.set_title(f"Learning Curves (degree={degree}, λ={alpha})")
        ax.grid(True)

        
        X_fit, y_fit, y_pred_train, y_pred_test = plot_fitting(degree,alpha)
      

        ax1.plot(X_fit, y_fit,linewidth = 3, label=f'λ={alpha}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title(f'Polynomial Ridge Regression Fit: deg={degree}')
        ax1.legend(loc = 'best')
        

deg = [0,5,10,15,30,100]
alpha = 10



ind = 0

fig, axes = plt.subplots(2,3, figsize=(14, 9))  


for i in range(2):
    for j in range(3):
        ax = axes[i, j]  
        degree = deg[ind]
        ind = ind + 1
        X_fit, y_fit, y_pred_train, y_pred_test = plot_fitting(degree,alpha)
      
        ax.scatter(X_train, y_train, label='Training Data', alpha=0.5)
        ax.scatter(X_val, y_val, label='Validation Data', alpha=0.5)
        ax.plot(X_fit, y_fit, label='Fit', color='k')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'Polynomial Ridge Regression Fit: deg={degree}')
        ax.legend(loc = 'best')
       
        


plt.tight_layout()
plt.show()



 

