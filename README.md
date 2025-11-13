# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas

2. Import Decision tree classifier

3. Fit the data in the model

4. Find the accuracy score

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Lorena Avelyn R
RegisterNumber: 212224040174
```
```python
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```

```python
print("data.info():")
data.info()
```
```python
print("isnull() and sum():")
data.isnull().sum()
```
```python
print("data value counts():")
data["left"].value_counts()
```
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```python
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```python
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```python
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```python
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```python
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:

<img width="1684" height="267" alt="image" src="https://github.com/user-attachments/assets/e75a5ddb-5d25-457b-a488-8e4d1f2a1176" />

<img width="1727" height="414" alt="image" src="https://github.com/user-attachments/assets/99a42319-e72b-46c9-9278-5e0d3a5f2e89" />

<img width="1779" height="531" alt="image" src="https://github.com/user-attachments/assets/afadddd1-6cce-4de1-86c7-a39cf15e95cc" />

<img width="1598" height="241" alt="image" src="https://github.com/user-attachments/assets/9c4271a8-08ec-490c-9f83-7b6139711bb4" />

<img width="1665" height="340" alt="image" src="https://github.com/user-attachments/assets/8ea1efe4-f0f1-4024-8e63-79e6531edf46" />

<img width="1646" height="318" alt="image" src="https://github.com/user-attachments/assets/fd5ace53-850e-47cc-8d17-4cda70e76a80" />

<img width="1753" height="67" alt="image" src="https://github.com/user-attachments/assets/1c5305bc-145a-4419-96f5-844902699489" />

<img width="1632" height="46" alt="image" src="https://github.com/user-attachments/assets/9443562b-2233-4b2e-9ff1-e7666f6cef9f" />

<img width="1739" height="666" alt="image" src="https://github.com/user-attachments/assets/eaefc892-7f90-4f4e-9d0e-b3716c99d152" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
