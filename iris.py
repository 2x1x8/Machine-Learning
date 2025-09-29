from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
w_test = np.array([[1,2,4,3],[1,1,3,1],[1,2,1,1]])
b_test = np.array([0,0,0])
class_size = 3
flower = ["setosa", "versicolor", "virginica"]
def one_hot(clas, size):
    arr = np.array([[0 for _ in range(size)] for j in range(len(clas))])
    for i in range(len(arr)):
        arr[i][clas[i]] = 1
    return arr
y_train = one_hot(y_train, class_size)
def softmax(x, w, b):
    return math.e**(x@(w.T) + b)/(np.sum(math.e**(x@w.T + b), 1)[:,None])
def compute_cost(x, y, w, b):
    return np.mean(-np.log(np.sum(y*softmax(x,w,b), 1)))
def dj_dw(x,y,w,b):
    return ((softmax(x,w,b).T - y.T)@x)/y.size
def dj_db(x,y,w,b):
    return np.mean((softmax(x,w,b).T - y.T), 1)
def grad_des(x, y, w, b, alpha, iter):
    for i in range(iter):
        w = w - alpha*dj_dw(x, y, w, b)
        b = b - alpha*dj_db(x, y, w, b)
    return w, b
def predict(x,w,b):
    
    y = softmax(x,w,b)
    ans = [0]*len(y)
    for i in range(len(y)):
        maxnum = 0
        for j in range(y[0].size):
            if y[i][j] > maxnum:
                maxnum = y[i][j]
                ans[i] = j
    return ans
w_ans, b_ans = grad_des(x_train, y_train, w_test, b_test, 1, 1000)
print(w_ans, b_ans)
print(compute_cost(x_train,y_train, w_ans, b_ans))
print(dj_dw(x_train,y_train, w_ans, b_ans))
print(predict(x_test, w_ans, b_ans))
print(y_test)