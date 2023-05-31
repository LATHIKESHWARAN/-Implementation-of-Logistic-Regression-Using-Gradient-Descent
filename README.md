# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Use the standard libraries in python for finding linear regression.
 2.Set variables for assigning dataset values. 
 3.Import linear regression from sklearn. 
 4.Predict the values of array.
 5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 6.Obtain the graph.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: LATHIKESHWARAN J
RegisterNumber:  212222230072
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data =np.loadtxt("/content/ex2data1(2).txt",delimiter=",")
x=data[:,[0,1]]
y=data[:,2]
x[:5]
y[:5]
plt.figure()
plt.scatter(x[y ==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("exam 1 score")
plt.ylabel("exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()

def costfunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta= np.array([0,0,0])
j,grad=costfunction(theta,x_train,y)
print(j)
print(grad)
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta= np.array([-24,0.2,0.2])
j,grad=costfunction(theta,x_train,y)
print(j)
print(grad)
def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]

  return j
def gradient(theta,x,y):
   h=sigmoid(np.dot(x,theta))
  
   grad=np.dot(x.T,h-y)/x.shape[0]
   return grad  
   
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta= np.array([0,0,0])
res=optimize.minimize(fun=cost, x0=theta , args=(x_train , y), method ="Newton-CG", jac=gradient)

print(res.fun)
print(res.x)
  
def plotDecisionBoundry(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()
  
prob =sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0], 1)) , x))
  prob =sigmoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
np.mean(predict(res.x,x)==y)  

```

## Output:
### Array Value of x:
![O1](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/3899ecfa-033b-445b-9e6a-38559424cfcb)
### Array Value of y
![O2](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/29ed5888-f261-4554-81a4-f9ba8d036161)
### Exam 1 - score graph
![O3](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/61b318aa-85b8-444f-856b-70fa9c4d9450)
### Sigmoid function graph
![O4](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/d830844b-4264-4735-99ae-0032a3bd92bf)
### X_train_grad value
![O5](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/49360f8b-fc70-40cf-a05d-e81e15551e51)
### Y_train_grad value
![O6](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/c09d746a-fd08-403e-a6e1-b7999ddcc3c1)
### Print res.x
![O7](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/4478f33c-9101-4ca8-9d8f-6fc8a51372aa)
### Decision boundary - graph for exam score
![O8](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/a4e18186-8d46-401c-860e-b49704b1e585)
### Proability value
![O9](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/cf07a3f6-7461-4c35-afab-b5c3a091572e)
### Prediction value of mean
![O10](https://github.com/LATHIKESHWARAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393556/48b2d08b-7638-424d-820c-c34b34f1f63d)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

