# imoprt libraries
import pandas as pd
import numpy as np
import matplotlib as plt 



# reading dataset

data = pd.read_csv('Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# preprocessing

from sklearn.impute import SimpleImputer
imputer =SimpleImputer()

imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


# encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder ,  OneHotEncoder
label_encoder_x =LabelEncoder()
x[ : , 0 ]=label_encoder_x.fit_transform(x[ : , 0 ])


# Apply OneHotEncoder to column 0 (the first column)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X = ct.fit_transform(x) 

label_encoder_y= LabelEncoder()
y=label_encoder_y.fit_transform(y)

#x=pd.get_dummies(x, columns=['Country'], dtype=float)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y ,test_size=0.2,random_state=0,)



from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)








