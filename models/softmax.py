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
flower = ["setosa", "versicolor", "virginica"]
class softmax:
    samples, features = x_data.shape
        self.class_size = 3
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
        return ((softmax(x,w,b).T - y.T)@x)/len(y)
    def dj_db(x,y,w,b):
        return np.mean((softmax(x,w,b).T - y.T), 1)
    def grad_des(x, y, alpha, iter):
        w = np.zeros((class_size ,features))
        b = np.zeros(class_size)
        for _ in range(iter):
            w = w - alpha*dj_dw(x, y, w, b)
            b = b - alpha*dj_db(x, y, w, b)
        return w, b
    def predict(x,w,b):
        
        y = softmax(x,w,b)
        ans = []
        for i in range(len(y)):
            maxnum = y[i][0]
            ans.append(0)
            for j in range(y[0].size):
                if y[i][j] > maxnum:
                    maxnum = y[i][j]
                    ans[i] = j
        return np.array(ans)
    w_ans, b_ans = grad_des(x_train, y_train, 1, 1000)
    y_pred = predict(x_data, w_ans, b_ans)
    print(w_ans, b_ans)
    print(compute_cost(x_train,y_train, w_ans, b_ans))
    print(dj_dw(x_train,y_train, w_ans, b_ans))
    print(np.mean(y_pred==y_data))
    print(y_test)
