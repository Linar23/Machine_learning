import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg

x1 = 7.5 + 2 * np.random.randn(100)
y1 = 5 + 2 * np.random.randn(100)

x2 = 2.5 + 2 * np.random.randn(100)
y2 = 15 + 2 * np.random.randn(100)

x3 = 12.5 + 2 * np.random.randn(100)
y3 = 15 + 2 * np.random.randn(100)

x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

plt.figure()
plt.title('Исходные данные')
plt.plot(x1,y1,'.r')
plt.plot(x2,y2,'.r')
plt.plot(x3,y3,'.r')

data_set = np.column_stack((x,y))

c1 = np.concatenate((10 + 2 * np.random.randn(1),5 + 2 * np.random.randn(1))) 
c2 = np.concatenate((2.5 + 2 * np.random.randn(1),15 + 2 * np.random.randn(1))) 
c3 = np.concatenate((12.5 + 2 * np.random.randn(1),15 + 2 * np.random.randn(1))) 

result = np.zeros(data_set[:,0].size)

for i in range(0,100):
    first = 0
    second = 0
    third = 0
        
    sum1_x = 0
    sum2_x = 0
    sum3_x = 0
    
    sum1_y = 0
    sum2_y = 0
    sum3_y = 0
    
    for i in range(0,data_set[:,0].size):
        min1 = linalg.norm(c1 - data_set[i])
        min2 = linalg.norm(c2 - data_set[i])
        min3 = linalg.norm(c3 - data_set[i])
        
        g = np.argmin([min1,min2,min3]) + 1
        result[i] = g
              
        if g == 1:
            first += 1
            sum1_x += data_set[i][0]
            sum1_y += data_set[i][1]
        elif g == 2:
            second += 1
            sum2_x += data_set[i][0]
            sum2_y += data_set[i][1]
        elif g == 3:
            third += 1
            sum3_x += data_set[i][0]
            sum3_y += data_set[i][1]
        
    c1 = [sum1_x / first,sum1_y / first]
    c2 = [sum2_x / second,sum2_y / second]
    c3 = [sum3_x / third,sum3_y / third]

cluster1_x = []
cluster1_y = []

cluster2_x = []
cluster2_y = []

cluster3_x = []
cluster3_y = []

for i in range(0,data_set[:,0].size):
    if result[i] == 1:
        cluster1_x.append(data_set[i][0])
        cluster1_y.append(data_set[i][1])
    elif result[i] == 2:
        cluster2_x.append(data_set[i][0])
        cluster2_y.append(data_set[i][1])
    elif result[i] == 3:
        cluster3_x.append(data_set[i][0])
        cluster3_y.append(data_set[i][1])

plt.figure()
plt.title('После кластеризации')
plt.plot(cluster1_x,cluster1_y,'.r')
plt.plot(cluster2_x,cluster2_y,'.g')
plt.plot(cluster3_x,cluster3_y,'.y')

# Центроиды
plt.plot(c1[0],c1[1],'x', markersize=10)
plt.plot(c2[0],c2[1],'x', markersize=10)
plt.plot(c3[0],c3[1],'x', markersize=10)