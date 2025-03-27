# Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data – Read Placement_Data.csv and remove unnecessary columns.

2.Convert Categorical Data – Encode categorical features into numerical values.

3.Prepare Data – Separate features (𝑋) and target variable (𝑌), initialize weights (𝜃).
   
4. Define Sigmoid Function – Compute probabilities using
   
    ![image](https://github.com/user-attachments/assets/e87535fe-a792-4573-ba32-4f3be1de1e5a)
   
5.Train Model – Update weights using gradient descent for multiple iterations.

![image](https://github.com/user-attachments/assets/76a24d55-fe01-493f-b1dd-4a643de2358a)

6.Make Predictions – Predict placement status (1 = Placed, 0 = Not Placed).

7.Evaluate & Test – Calculate accuracy and predict placement for new students.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.
```
Developed by: LATHIKESHWARAN J
RegisterNumber:  212222230072
```
```PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:

### Dataset:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/f38e0542-e307-4996-85f6-ca76711b8dd6" width=50%>

### Dataset.dtypes:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/c7a569d8-b4c5-4ef7-83e6-57affdc6b0f7" width=50%>

### Labeled_dataset:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/e7b4981d-56b7-4733-9ad0-e95a286aec97" width=50%>

### Dependent variable Y:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/714c4c29-63aa-4857-9911-4a04bcfb1835" width=50%>


### Accuracy:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/f070b9c1-43e2-4e7b-9ddd-a504e3d877cd">

### y_pred:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/cc07c9d6-9694-434b-875d-6e7ae8fa424d" width=50%>

## Y:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/f5633eb6-bb34-4944-b705-c79fa0d9a1b2" width=50%>

### y_pred:
<img src="https://github.com/Janarthanan2/ML_Ex05-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393515/19c2981a-ff8a-4fac-adca-16fb93d9890b" width=50%>

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
