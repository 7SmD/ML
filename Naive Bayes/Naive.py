import pandas as pd
df = pd.read_csv("titanic.csv")
print(df.head())

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

inputs = df.drop('Survived',axis='columns')
target = df.Survived

#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
dummies = pd.get_dummies(inputs.Sex)
print(dummies.head(3))

inputs = pd.concat([inputs,dummies],axis='columns')
print(inputs.head(3))

#I am dropping male column as well because of dummy variable trap theory. One column is enough to repressent male vs female

inputs.drop(['Sex','male'],axis='columns',inplace=True)
print(inputs.head(3))

print(inputs.columns[inputs.isna().any()])

print(inputs.Age[:10])

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)

#For Probability
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

print(X_test[0:10])

print(y_test[0:10])

print(model.predict(X_test[0:10]))

print(model.predict_proba(X_test[:10]))

#Calculate the score using cross validation

from sklearn.model_selection import cross_val_score
print(cross_val_score(GaussianNB(),X_train, y_train, cv=5))