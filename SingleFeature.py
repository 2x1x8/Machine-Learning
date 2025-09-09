import math, copy
import numpy as np
import matplotlib.pyplot as plt
HoursStudied = np.array([1.5, 3.0, 4.5, 6.0, 7.5])
TestScore = np.array([55, 65, 75, 85, 95])
x_train = np.array([1500, 2000, 2500, 3000, 3500])
y_train = np.array([300000, 350000, 400000, 450000, 500000])
x_test = np.array([6.0, 7.5])
y_test = np.array([85, 95])
def compute_cost(x, y, w, b):
  m = x.shape[0]  
  cost = 0
  for i in range(m):
    f_wb = w*x[i] + b
    cost += ((f_wb - y[i])**2)/2
  return cost
def dj_dw(x, y, w, b):
  m = x.shape[0]  
  dj_dw = 0
  for i in range(m):
    f_wb = w*x[i] + b
    dj_dw += (f_wb - y[i])*x[i]
  return dj_dw
def dj_db(x, y, w, b):
  m = x.shape[0]  
  dj_db = 0
  for i in range(m):
    f_wb = w*x[i] + b
    dj_db += (f_wb - y[i])
  return dj_db
def grad_des(x, y, w, b, alpha, iter):
  x = x / 1000
  y = y / 1000
  j_history = []
  p_history = []
  for i in range(iter):
    w = w - alpha*dj_dw(x,y,w,b)    
    b = b - alpha*dj_db(x,y,w,b)  
    j_history.append(compute_cost(x, y, w , b))
    p_history.append([w,b])
  return w*1000,b*1000, j_history, p_history

w_final, b_final, J_hist, p_hist = grad_des(x_train, y_train, 0, 0, 0.01, 10000)
print(w_final, "and", b_final) 
print(compute_cost(x_train, y_train, w_final, b_final))
fig, (ax1) = plt.subplots(1, 1, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist)
ax1.set_title("Cost vs. iteration(start)")
ax1.set_ylabel('Cost')      
ax1.set_xlabel('iteration step')  
plt.show()
