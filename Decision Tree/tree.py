import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("salaries.csv")
print(df.head())
print('\n')

inputs = df.drop('salary_more_then_100k', axis='columns')                     # X AXIS
target = df['salary_more_then_100k']                                         # Y AXIS

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
print(inputs)
print('\n')

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
print(inputs_n)
print('\n')

print(target)
print('\n')

#TREE
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)


sc = model.score(inputs_n, target)
print(sc)
print('\n')

#Is salary of Google, Computer Engineer, Bachelors degree > 100 k ?

print(model.predict([[2, 1, 0]]))

#Is salary of Google, Computer Engineer, Masters degree > 100 k ?

print(model.predict([[2, 1, 1]]))
