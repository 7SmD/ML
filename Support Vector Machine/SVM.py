import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

print(iris.feature_names)
print('\n')

print(iris.target_names)
print('\n')

df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
print('\n')

df['target'] = iris.target
print(df.head())
print('\n')

print(df[df.target==1].head())
print('\n')

print(df[df.target==2].head())
print('\n')

df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])                  # IMP. used for Conversion
print(df.head())
print('\n')

print(df[45:55])
print('\n')

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

import matplotlib.pyplot as plt

#Sepal length vs Sepal Width (Setosa vs Versicolor)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
plt.show()

#Train Using Support Vector Machine (SVM)

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train))
print('\n')

print(len(X_test))
print('\n')

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(model.predict([[4.8,3.0,1.5,0.3]]))

