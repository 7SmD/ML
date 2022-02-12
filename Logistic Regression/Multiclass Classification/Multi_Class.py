from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()

#DESCRIPTION
print(dir(digits))
print('\n')

print(digits.data[0])

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    #plt.show()

print('\n')

#Create and train logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)
model.fit(X_train, y_train)

#Measure accuracy of our model
sc=model.score(X_test, y_test)
print(sc)
print('\n')

print(model.predict(digits.data[0:5]))
print('\n')

#Confusion matrix
y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()