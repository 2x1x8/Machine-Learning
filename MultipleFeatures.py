import copy
import numpy as np
import matplotlib.pyplot as plt

x_raw = np.array([[50000, 3, 1600],   # Car 1
                  [70000, 5, 1800],   # Car 2
                  [30000, 2, 1400],   # Car 3
                  [100000, 8, 2000],  # Car 4
                  [40000, 1, 1600]])
y_raw = np.array([29000, 27000, 29500, 23000, 32500])
def standardize(x):
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

def compute_cost(x, y, w, b):
  m, n = x.shape 
  f_wb = x@w + b 
  return np.sum((f_wb - y)**2)/(m*2)
def dj_dw(x, y, w, b):
  m,n = x.shape 
  f_wb = x @ w + b
  return ((f_wb - y)/m)@x.T
def dj_db(x, y, w, b):
  m,n = x.shape  
  f_wb = x@w + b
  return np.sum(f_wb - y)/m
def grad_des(x_raw, y_raw, w, b, alpha, iter):
  x = standardize(x_raw)
  y = standardize(y_raw)
  for i in range(iter):
    w = w - alpha*dj_dw(x,y,w,b)
    b = b - alpha*dj_db(x,y,w,b) 
  return unstandardize(w, b, x_raw, y_raw)

w_final, b_final = grad_des(x_raw, y_raw, np.array([-0.05,0.188,-0.001]), 0, 0.6, 10000)
print("w:", w_final, "and", "b:", b_final)
print(dj_dw(x_raw, y_raw, w_final,b_final))
print(dj_db(x_raw, y_raw, w_final,b_final))
print(compute_cost(x_raw, y_raw, w_final,b_final))
print("correct value: w = [0.0705,0.2396,0.0402], b = 0.142")
