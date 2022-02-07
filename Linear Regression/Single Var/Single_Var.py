import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#reading csv file and printing it
df = pd.read_csv("LR_SV.csv")
print(df)

#Fitting training data in model
reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

# " y = mx + b " => isme m is coffecient and => x is independent var(FEATURES) => y is dependent var => b is intercept
print("\n")
m=reg.coef_
b=reg.intercept_
print("Coeffecients : ",m)
print("Intercept : ",b)

#predicting new val Acc to req
print("\n")
prediction=reg.predict([[3300]])
print("Predicted val : ",prediction)

#plotting
plt.scatter(df.area,df.price)
plt.plot(df.area,reg.predict(df[['area']]))
plt.show()