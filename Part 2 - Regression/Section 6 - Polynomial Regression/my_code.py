#importing libraries 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# reading data 

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:2]
Y =data.iloc[: , -1]

##no spliting of data because the data is to small
#linear regression 
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X,Y)

#plynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_model = PolynomialFeatures(degree=4)
poly=poly_model.fit_transform(X,Y)
linear_model_2 = LinearRegression()
linear_model_2.fit(poly,Y)


#visalization 

plt.scatter(X, Y, color='red')
plt.plot(X,linear_model.predict(X),color='blue')
plt.plot(X,linear_model_2.predict(poly),color='black')

plt.legend()
plt.show()


#prdict new employee

linear_model_2.predict(poly_model.fit_transform([[6.5]]))








