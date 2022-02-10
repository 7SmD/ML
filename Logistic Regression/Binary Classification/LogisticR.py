import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("insurance_data.csv")
head=df.head()                                    #returns top 5
print(head)
print('\n')

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
print(X_test)
print('\n')


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(X_test)
print('\n')

y_predicted = model.predict(X_test)
print(y_predicted)
print('\n')

ans=model.predict_proba(X_test)
print(ans)
print('\n')

score=model.score(X_test,y_test)
print(score)
print('\n')

print(y_predicted)

