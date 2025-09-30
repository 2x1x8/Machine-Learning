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
    random_state=50 # for reproducibility
)
def knn(x, x_data, y_data,k):
    dist = np.sqrt(np.sum(np.square(x_data), 1).reshape(1,-1)
                    + np.sum(np.square(x), 1).reshape(-1,1)
                    - 2*(x@x_data.T)
    )
    predictions = np.empty(len(x), dtype=y_data.dtype)
    for i in range(len(x)):
        neighbors_idx = np.argpartition(dist[i], k)[:k]  # faster than argsort
        neighbor_labels = y_data[neighbors_idx]

        # majority vote
        if np.issubdtype(y_data.dtype, np.integer):
            counts = np.bincount(neighbor_labels)
            predictions[i] = np.argmax(counts)
        else:
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            predictions[i] = labels[np.argmax(counts)]
    return predictions
    
print( y_test)
print(knn(x_test, x_train, y_train,10))

        