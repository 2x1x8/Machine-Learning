from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math as math
iris = load_iris()
x_data = np.array(iris.data)   # features (sepal length/width, petal length/width)
y_data = np.array(iris.target) # labels (0=setosa, 1=versicolor, 2=virginica)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, 
    test_size=0.2,   # 20% test, 80% train
    random_state=42 # for reproducibility
)
classes = np.unique(y_data)
w_test = np.array([0,0,0,0])
b_test = 0
def convert(y, a):
    return np.where(y==a,1,-1)
def standardize(x):
  mu = np.mean(x, axis=0)
  sigma = np.std(x, axis=0)
  x = (x - mu) / sigma
  return x
def unstandardize(w, b, x_old):
  mu_x = np.mean(x_old, axis=0)
  sigma_x = np.std(x_old, axis=0)
  ans_w = w/sigma_x
  ans_b = b
  ans_b -= np.sum(mu_x*w/sigma_x)
  return ans_w, ans_b
def hinge_loss(w, b, x, y,c):
    return 0.5*np.sum(w**2) + c*np.mean(np.maximum(0, 1 - y*(x@w + b)))
def dj_dw(w, b, x, y, c):
    return w + c*(np.where(1 - y*(x@w + b)<= 0, 0, -y)@x)/len(x)
def dj_db(w, b, x, y, c):
    return c*np.mean(np.where(1 - y*(x@w + b) <= 0, 0, -y))
def fit(x, y, c, alpha, iter):
    w = np.array([0]*x.shape[1])
    b = 0

    for i in range(iter):
        w = w - alpha*dj_dw(w, b, x, y, c)
        b = b - alpha*dj_db(w, b, x, y, c)
    return w, b

def predict(w, b, x):
    return np.argmax(x@w.T + b, axis=1)
wAns = []
bAns = []
if len(classes) == 2:
    wAns, bAns = (fit(x_train,convert(y_train, classes[0]),100,0.01,1000))
else:
    for i in classes:
        w_i, b_i = (fit(x_train,convert(y_train, i),100,0.01,1000))
        wAns.append(w_i)
        bAns.append(b_i)
wAns, bAns = np.array(wAns), np.array(bAns)
print(wAns)
print(hinge_loss(wAns[0], bAns[0], x_train, y_train,100))
print(y_test)
print(predict(wAns, bAns, x_test))
print(y_test - predict(wAns, bAns, x_test))
y_pred = predict(wAns, bAns, x_test)
acc = np.mean(y_pred == y_test)
print("Test accuracy:", acc)