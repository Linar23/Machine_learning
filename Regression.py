from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1, 1000) # Обучающая выборка
y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x) # Некоторая функция
error = 10 * np.random.randn(1000) # Ошибка
t = y + error # Целевые значения - некоторая функция + шум
t = t.reshape((1000, 1))

# w = (Фтрансп. * Ф)^(-1) * Фтрансп. * t)
def computing_w(F):
    w = F.transpose().dot(F)
    w = linalg.inv(w)
    w = w.dot(F.transpose())
    w = w.dot(t)
    return w

def computing_F_polynomial(x,n):
    F=np.ones((x.size, n+1))
    for i in range(1,n+1):
        F[:,i]=x**i
    return F

# Линейная регрессия
vector_ones = np.ones(1000)
F = np.float32([vector_ones,x]).transpose()
w = computing_w(F)
z = F.dot(w)

# График
plt.figure() # Формируем графическое окно
plt.title('Линейная регрессия')

plt.plot(x, t, '.g') # Целевые значения
plt.plot(x, y , 'r') # Истинный тренд
plt.plot(x, z, 'b') # Предсказанные значения

# Полиномиальная регрессия
F_polynomial = computing_F_polynomial(x,9) # При этой степени ошибка наименьшая
w_polynomial = computing_w(F_polynomial)
z_polynomial = F_polynomial.dot(w_polynomial)

# График
plt.figure() # Формируем графическое окно
plt.title('Полиномиальная регрессия')

plt.plot(x, t, '.g') # Целевые значения
plt.plot(x, y , 'r') # Истинный тренд
plt.plot(x, z_polynomial, 'b') # Предсказанные значения

# График зависимости ошибки от степени M
error_sum = []

for i in range(2,20):
    F_polynomial = computing_F_polynomial(x,i)
    w_polynomial = computing_w(F_polynomial)
    z_polynomial = F_polynomial.dot(w_polynomial)
    error_sum.append(.5 * np.sum((z_polynomial - t)**2))
    
polynom_powers = np.arange(2,20) # Степени полинома

plt.figure()
plt.title('График зависимости ошибки от степени M') 
plt.plot(polynom_powers,error_sum)