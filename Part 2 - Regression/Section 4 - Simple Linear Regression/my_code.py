#importing libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt



#reading dataset
data=pd.read_csv('Salary_Data.csv')
X=data.iloc[:,0]
Y=data.iloc[:,1]

#prepocessing data

from sklearn.model_selection import train_test_split
x_train , x_test ,y_train,y_test=train_test_split(X,Y, test_size=0.2,random_state=0)

"""
from sklearn.preprocessing import StandardScaler
sd_x=StandardScaler()
x_train=sd_x.fit_transform(x_train)
x_test=sd_x.transform(x_test)
y_train=sd_x.transform(y_train)
y_test=sd_x.transform(y_test)
"""
#build model

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(pd.DataFrame(x_train),y_train)

#predict data
y_pred = model.predict(pd.DataFrame(x_test))
y_pred_train = model.predict(pd.DataFrame(x_train))


plt.scatter(x_train, y_train, color='red', label='Training data')
plt.plot(x_train, y_pred_train, color='blue', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience (Training set)')
plt.legend()
plt.show(block=True)


plt.scatter(x_test, y_test, color='red', label='Training data')
plt.plot(x_test, y_pred, color='blue', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience (Training set)')
plt.legend()
plt.show(block=True)



