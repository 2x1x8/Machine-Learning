from sklearn.datasets import load_iris
import math as math
iris = load_iris()
x_data = iris.data   # features (sepal length/width, petal length/width)
y_data = iris.target # labels (0=setosa, 1=versicolor, 2=virginica)
print(y_data)
flower = ["setosa", "versicolor", "virginica"]
def softmax(x, w, b, c):
    return math.e**(x[:,c]@w + b)/(math.e**(x[c]@w + b))
