import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

#  load the dataset

df= pd.read_csv("calefornial_housing.csv")
#Details of dataset
print(df.head())
print(df.columns)
print(df.shape)
print(df.info)
print(df.describe)


#seprate feature and target 

X= df.drop("target",axis=1)

y=df["target"]


#seprate the dataset into traning and testing 

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


#model**************

model=LinearRegression()


##  Train  The Model


model.fit(X_train,y_train)


#test the data

y_pred=model.predict(X_test)

##  Model Evalution 

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Evalution of model")
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared error==",mse)
mae= mean_absolute_error(y_test,y_pred)
print("mean absolute error==",mae)
r_square=r2_score(y_test,y_pred)
print("Rscore==",r_square)


