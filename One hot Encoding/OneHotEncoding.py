import pandas as pd
df = pd.read_csv("carprices.csv")
print(df)
print('\n')

#creating dummy var for carmodels
dummies = pd.get_dummies(df['Car Model'])
print(dummies)
print('\n')

#merging 2 tables
merged = pd.concat([df,dummies],axis='columns')
print(merged)
print('\n')

#droping/removing sm columns
final = merged.drop(["Car Model","Mercedez Benz C class"],axis='columns')
print(final)
print('\n')

# x=bchi hui val
# y=price
X = final.drop('Sell Price($)',axis='columns')
print(X)
print('\n')

y = final['Sell Price($)']
print(y)
print('\n')

#selecting model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)

#printing accuracy of model
print(model.score(X,y))
print('\n')

#Price of mercedez benz that is 4 yr old with mileage 45000
benz=model.predict([[45000,4,0,0]])                             #2d arr
print(benz)
#Price of BMW X5 that is 7 yr old with mileage 86000
bmw=model.predict([[86000,7,0,1]])
print(bmw)

