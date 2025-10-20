# ==============================================
# IMPORT LIBRARIES
# ==============================================
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ==============================================
# READ DATA
# ==============================================
data = pd.read_csv('50_Startups.csv')

X = data.iloc[:, 0:4].values
Y = data.iloc[:, -1].values

# ==============================================
# ENCODING CATEGORICAL VARIABLE (STATE)
# ==============================================
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Avoid dummy variable trap
X = X[:, 1:]

# ==============================================
# SPLIT DATA
# ==============================================
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# ==============================================
# MODEL 1: LINEAR REGRESSION (ALL FEATURES)
# ==============================================
model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)

# ==============================================
# BACKWARD ELIMINATION USING STATS MODELS
# ==============================================
X_with_intercept = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
x_opt = X_with_intercept[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=x_opt).fit()
print(regressor_OLS.summary())

# Remove highest p-value features step by step (you found best = 3 and 5)
x_opt_final = X_with_intercept[:, [0,3,5]]

# ==============================================
# MODEL 2: LINEAR REGRESSION (SELECTED FEATURES)
# ==============================================
# Note: we must extract the same columns from the *train/test sets* as used in final x_opt
X_opt_train = X_train[:, [2, 4]]  # since your X had no intercept, map [3,5] → [2,4]
X_opt_test  = X_test[:, [2, 4]]

model_opt = LinearRegression()
model_opt.fit(X_opt_train, y_train)
y_pred_opt = model_opt.predict(X_opt_test)

# ==============================================
# EVALUATE BOTH MODELS
# ==============================================
def evaluate_model(y_true, y_pred, name):
    print(f"📊 {name}")
    print(f"MAE  = {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE  = {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²   = {r2_score(y_true, y_pred):.4f}")
    print("-" * 40)

evaluate_model(y_test, y_pred_all, "Model 1: All Features")
evaluate_model(y_test, y_pred_opt, "Model 2: After Backward Elimination")

# ==============================================
# VISUAL COMPARISON
# ==============================================
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_all, alpha=0.5, label='All Features')
plt.scatter(y_test, y_pred_opt, alpha=0.5, label='Selected Features')
plt.plot(y_test, y_test, color='red', label='Perfect Prediction')
plt.xlabel("True Profit")
plt.ylabel("Predicted Profit")
plt.legend()
plt.grid(True)
plt.title("Model Comparison: All vs Optimal Features")
plt.show()
