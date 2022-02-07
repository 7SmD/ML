import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#reading csv file and printing it
df = pd.read_csv("LR_MV.csv")
print(df)

#Hmne dekha ki bedroom me nan value h so usko replace kr denge by median
import math
median_bedroom=math.floor(df.bedroom.median())
print("\nMedian : ",median_bedroom)

#filling NAN values
df.bedroom=df.bedroom.fillna(median_bedroom)
print("\n",df)

#Fitting training data in model
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)

# " y = m1x1 + m2x2 + ....... + mnxn + b " => isme m is coffecient and => x is independent var(FEATURES) => y is dependent var => b is intercept
print("\n")
m=reg.coef_
b=reg.intercept_
print("Coeffecients : ",m)
print("Intercept : ",b)

#predicting new val Acc to req
print("\n")
prediction=reg.predict([[3000,3,15]])
print("Predicted val : ",prediction)

