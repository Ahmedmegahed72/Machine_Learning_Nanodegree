#importing library 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt




# ==============================================
# READ DATA
# ==============================================
data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:, 1:2].values
Y = data.iloc[:, -1].values




#scr need feature scaling
from sklearn.preprocessing import StandardScaler
scaler_x=StandardScaler()
scaler_y=StandardScaler()

X=scaler_x.fit_transform(X)
Y=np.ravel(scaler_y.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR
model = SVR(kernel='rbf')
model.fit(X,Y)

y_pred=scaler_x.transform(np.array([[6.5]]))
y_pred=model.predict(y_pred)
y_pred=scaler_y.inverse_transform(y_pred.reshape(-1,1))



plt.figure(figsize=(6, 4))
plt.scatter(X,Y , alpha=0.5, label='svr')
plt.plot(X,model.predict(X),color='red')
plt.xlabel("True Profit")
plt.ylabel("Predicted Profit")
plt.legend()
plt.grid(True)
plt.show()





