from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1, 1000)

ind = np.arange(1000)
np.random.shuffle(ind)

x_1 = x[ind[:500]]
y_1 = 20 * np.sin(2 * np.pi * 3 * x_1) + 100 * np.exp(x_1) # Некоторая функция
error = 10 * np.random.randn(500) # Ошибка
t_1 = y_1 + error

# Добавляем класс
class_0 = np.zeros(shape=(t_1.size))
x_1_with_class = np.vstack((x_1, class_0)).transpose()
t_1_with_class = np.vstack((t_1, class_0)).transpose()

x_2 = x[ind[500:1000]]
y_2 = 20 * np.sin(2 * np.pi * 3 * x_2) + 100 * np.exp(x_2) + 100 # Некоторая функция
error = 10 * np.random.randn(500) # Ошибка
t_2 = y_2 + error

# Добавляем класс
class_1 = np.ones(shape=(t_2.size))
x_2_with_class = np.vstack((x_2, class_1)).transpose()
t_2_with_class = np.vstack((t_2, class_1)).transpose()

x_data_set = np.vstack((x_1_with_class, x_2_with_class))
y = np.vstack((t_1_with_class, t_2_with_class))

# Разделяем выборку
ind = np.arange(1000)
np.random.shuffle(ind)

x_train = x_data_set[ind[:600]]
x_val = x_data_set[ind[600:800]]
x_test = x_data_set[ind[800:]]

y_train = y[ind[:600]]
y_val = y[ind[600:800]]
y_test = y[ind[800:]]

def computing_fert(x,m):
    fert = np.ones((x.size, m+1))
    for i in range(1,m+1):
        fert[:,i] = x ** i
            
    return fert.transpose()

def sigmoid(w,fert):
    exp = np.power(np.e,-w.dot(fert))
    sig = 1 / (1 + exp)
    
    return sig.transpose()

def gradient(N,t,fert,w,lam):
    inter = 0
    inter_sigmoid = sigmoid(w,fert)
    for i in range(1,N):
        inter += (inter_sigmoid[i] - t[i]) * fert[:,i]
    grad = inter + lam * w
                      
    return grad
   
def gradient_descent(N,t,fert):
    w_0 = np.matrix([[1,1,1,1,1,1,1]])
    w_k_1 = w_0
    w_k = w_k_1 - 0.1 * gradient(N,t,fert,w_k_1,0.1)
    while linalg.norm(w_k - w_k_1) <= 0.001 * (linalg.norm(w_k) + 0.001):
        w_k_1 = w_k
        w_k = w_k_1 - 0.1 * gradient(N,t,fert,w_k_1,0)
        
    return w_k
    
test_fert = computing_fert(x_train[:,:1].transpose(),6)

#w_0 = np.matrix([[1,1,1,1,1]])
#test_sigmoid = sigmoid(w_0,test_fert)
#
#test_gradient = gradient(600,x_train[:,1:2],test_fert,w_0,0)
test_gradient_descent = gradient_descent(600,x_train[:,1:2],test_fert)

y_predicted = test_gradient_descent.dot(test_fert)
result = sigmoid(test_gradient_descent,test_fert) * 10000000000