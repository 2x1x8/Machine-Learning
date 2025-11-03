import math, copy
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([
   [ 3.53,  0.80],
   [ 1.96,  4.48],
   [ 3.74, -1.95],
   [ 1.90, -0.30],
   [-0.21,  0.82]   # Likely class 0
])

# Labels (y): shape (6,)
y_data = np.array([1, 0, 1, 1, 0])


def standardize(x_raw):
  x = copy.deepcopy(x_raw)
  mu = np.mean(x, axis=0)
  sigma = np.std(x, axis=0)
  x = (x - mu) / sigma
  return x
def unstandardize(w, b, x_old, y_old):
  mu_x = np.mean(x_old, axis=0)
  sigma_x = np.std(x_old, axis=0)
  mu_y = np.mean(y_old)
  sigma_y = np.std(y_old)
  ans_w = (w*sigma_y)/sigma_x
  ans_b = b
  ans_b -= np.sum(mu_x*w/sigma_x)
  ans_b = ans_b*sigma_y + mu_y
  return ans_w, ans_b

def cost(x, y, w, b):
  sigmoid = 1/(1 + math.e**(-(x@w + b)))
  loss = np.log(sigmoid)*y + (1-y)*np.log(1-sigmoid)
  return -np.mean(loss)
def dj_dw(x, y, w, b): 
  sigmoid = 1/(1 + math.e**(-(x@w + b)))
  return (x.T@(sigmoid-y))/y.shape[0]
def dj_db(x, y, w, b):
  sigmoid = 1/(1 + math.e**(-(x@w + b)))
  return np.mean(sigmoid-y)
def grad_des(x_raw, y, w, b, alpha, iter):
  x = standardize(x_raw)
  for i in range(iter):
    w = w - alpha*dj_dw(x, y, w, b)
    b = b - alpha*dj_db(x, y, w, b)
  return unstandardize(w, b, x_raw, y)
w_final, b_final = grad_des(x_data, y_data, np.array([0, 0]), 0, 10, 100)
print("w:", w_final,"b:", b_final)
print(cost(x_data, y_data, w_final, b_final))
print("correct w:(), b:")
print(x_data, b_final,w_final)
print(1/(1 + math.e**(-(x_data@w_final + b_final))))

plt.plot(x_data[:, 0], (-b_final - x_data[:,0]*w_final[0])/w_final[1] , c='b',label='Our Prediction')
plt.scatter(x_data[:, 0], x_data[:, 1])  # Only the first feature
plt.show()
plt.xlabel('b')
plt.ylabel("a")
