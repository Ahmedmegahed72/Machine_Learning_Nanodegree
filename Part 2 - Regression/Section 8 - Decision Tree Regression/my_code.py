# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# ==============================================
# EVALUATE BOTH MODELS
# ==============================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate_model(y_true, y_pred, name):
    print(f"📊 {name}")
    print(f"MAE  = {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE  = {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²   = {r2_score(y_true, y_pred):.4f}")
    print("-" * 40)

evaluate_model(y, regressor.predict(X), "Model 1: All Features")

