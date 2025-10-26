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
samples, features = x_data.shape
x_data_cent = x_data - np.mean(x_data, axis = 0)
covar = (x_data_cent.T@x_data_cent)/(x_data_cent.shape[0]-1)
eval, evec = np.linalg.eig(covar)
idx = np.argsort(eval)[::-1]
evec = evec[:,idx]
pca = x_data_cent@evec
pairs = [(0,1), (0,2), (1,2)]
fig, axs = plt.subplots(1, len(pairs), figsize=(4*len(pairs), 4))
for a, (i, j) in enumerate(pairs):
    axs[a].scatter(pca[:, i], pca[:, j], c=y_data)
    axs[a].set_xlabel(f"PC{i+1}")
    axs[a].set_ylabel(f"PC{j+1}")
    axs[a].set_title(f"PC{i+1} vs PC{j+1}")

plt.tight_layout()  # avoids overlapping labels
plt.show()