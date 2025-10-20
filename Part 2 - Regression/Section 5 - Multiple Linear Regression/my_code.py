"""ou’ve built two regression models:

A normal linear regression (using all features).

A statistical feature-selected regression (using only the best features found by backward elimination).

Now you want to compare their performance using error metrics such as:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score (Coefficient of Determination)"""

# immporting libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Reading dataset

Data = pd.read_csv('50_Startups.csv')
X = Data.iloc[: ,0:4 ]
Y = Data.iloc[:,-1]
#preprocessing

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#avoid dumy variable trap
X=X[:,1:]



from sklearn.model_selection import train_test_split
x_train ,x_test , y_train , y_test = train_test_split(X,Y, test_size=0.2,random_state=0)



#build model
from sklearn.linear_model import LinearRegression
model =  LinearRegression()
model.fit(x_train,y_train)


#predict result 
y_pred=model.predict(x_test)


## build the optional model using backward elimination

import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
x_opt=X[:,[0,1,2,3,4,5]]
regressor=sm.OLS(endog=Y, exog=x_opt).fit()

regressor.summary()

"""
#remove col 2 becouse it have higher p_value
x_opt=X[:,[0,1,3,4,5]]
regressor=sm.OLS(endog=Y, exog=x_opt).fit()
regressor.summary()

#هتفضل تشيل اعلي كولم حقق اعلي ب فاليو
x_opt=X[:,[0,3,5]]
regressor=sm.OLS(endog=Y, exog=x_opt).fit()
regressor.summary()
nterpreting this

The column 0 is the intercept term (the added column of ones) — keep it always.

Columns 3 and 5 are the features that remain statistically significant (p-value ≤ 0.05)

"""
# best model based on feature selection and statistical
from sklearn.linear_model import LinearRegression

# Extract only the significant features (3 and 5)
X_optimal = x_test[:, [2,4]]

regressor_final = LinearRegression()
regressor_final.fit(X_optimal, y_test)

# Prediction example
y_pred_optimal = regressor_final.predict(X_optimal)
#visulization
plt.figure(figsize=(6, 4))
plt.scatter(Y, y_pred_optimal, alpha=0.4)
plt.plot(Y,Y, c='red')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.grid(axis='both')
plt.show()

