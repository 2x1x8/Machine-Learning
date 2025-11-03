from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from qpsolvers import solve_qp
from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
import math as math
x_data, y_data = make_circles(n_samples=100, factor=0.4, noise=0.1, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, 
    test_size=0.2,   # 20% test, 80% train
    random_state=42 # for reproducibility
)
PCA().plot(x_data,y_data)
classes = np.unique(y_data)
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
def rbf_kernel(X1, X2, gamma=0.5):
    sq_dists = np.sum(X1**2, axis=1)[:,None] + np.sum(X2**2, axis=1)[None,:] - 2*X1@X2.T
    return np.exp(-gamma * sq_dists)
def train_dual_svm(X, y, C=1.0):
    n = len(y)
    K = rbf_kernel(X, X, gamma=0.5)
    K = (K+K.T)/2+ np.eye(n)*1e-8
    P = np.outer(y, y) * K
    P = (P + P.T) / 2 + np.eye(n) * 1e-4
    q = -np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n),C*np.ones(n)])
    A = y.reshape(1, -1)
    print('e')
    print(A.shape)
    b = np.array([0.])
    print("n =", n)
    print("P shape:", P.shape)
    print("PD check:", np.min(np.linalg.eigvalsh(P)))
    print("y sum:", np.sum(y))
    print("unique y:", np.unique(y))
    a = solve_qp(P, q, G, h, A, b, solver="daqp")  # or "proxqp"
    sv = (a > 1e-6) & (a < C - 1e-6) 
    if np.any(sv):
        b = np.mean(y[sv] - np.sum(((a * y)[:, None] * rbf_kernel(X, X[sv])), axis=0))
    else:
        b = 0.0
    return a, b

def decision_function(X_train, y_train, a, b, X_test, kernel):
    K = kernel(X_train, X_test)  # shape (n_train, n_test)   
    f = np.sum(((a * y_train)[:, None] * K), axis = 0)+ b
    return f

def predict(X_train, y_train, a, b, X_test, kernel):
    f = decision_function(X_train, y_train, a, b, X_test, kernel)
    print(f)
    return np.sign(f)
a, b = train_dual_svm(x_train, convert(y_train, 0), C=10.0)

y_pred = predict(x_train,convert(y_train, 0),a,b , x_test, rbf_kernel)
y_test = convert(y_test, 0)
print(y_test)
print(y_pred)
print(y_test - y_pred)

acc = np.mean(y_pred == y_test)
print("Test accuracy:", acc)

x1 = np.linspace(min(x_data[:, 0]), max(x_data[:, 0]), 400)
x2 = np.linspace(min(x_data[:, 1]), max(x_data[:, 1]), 400)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]   # shape (160000, 2)
F = decision_function(x_train, convert(y_train, 0), a, b, X_grid, rbf_kernel)
F = F.reshape(X1.shape)
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
plt.contourf(X1, X2, F,  cmap = 'jet')
plt.colorbar()
axs.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
axs.set_xlabel(f"X1")
axs.set_ylabel(f"X2")
axs.set_title(f"moons")
plt.tight_layout()  # avoids overlapping labels
plt.show()
