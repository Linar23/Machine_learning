import numpy as np

def alpha_error(FP,TN):
    return FP/(TN + FP)

def beta_error(FN,TP):
    return FN/(TP + FN)

def accuracy(TP,TN,N):
    return (TP + TN) / N

def precision(TP,FP):
    return TP / (TP + FP)

def recall(TP,FN):
    return TP/(TP + FN)

def F1_Score(Pr,Rec):
    return 2 * (Pr * Rec) / (Pr + Rec)

footballers = 10 * np.random.randn(500) + 190                          
basketball_players = 10 * np.random.randn(500) + 210
                             
coin = np.random.binomial(1, .5 , 1000)

table_coin = [[0,0],[0,0]]
table_limit = [[0,0],[0,0]]

# 1 ый способ - монетка
for i in range(0,500):
    if coin[i] == 0:
        table_coin[0][0] += 1
    else:
        table_coin[0][1] += 1

for i in range(500,1000):
    if coin[i] == 0:
        table_coin[1][0] += 1
    else:
        table_coin[1][1] += 1

# 2 ой способ - порог   
for i in range(0,500):
    if footballers[i] <= 195:
        table_limit[0][0] += 1
    else:
        table_limit[0][1] += 1

for i in range(0,500):
    if basketball_players[i] <= 195:
        table_limit[1][0] += 1
    else:
        table_limit[1][1] += 1
                 
alpha_error_coin = alpha_error(table_coin[1][0],table_coin[0][0])
alpha_error_limit = alpha_error(table_limit[1][0],table_limit[0][0])

print('Ошибка первого рода для монетки = ',alpha_error_coin)
print('Ошибка первого рода для порога = ',alpha_error_limit)
print()

accuracy_coin = accuracy(table_coin[1][1],table_coin[0][0],footballers.size + basketball_players.size)
accuracy_limit = accuracy(table_limit[1][1],table_limit[0][0],footballers.size + basketball_players.size)

print('Accuracy для монетки = ', accuracy_coin)
print('Accuracy для порога = ', accuracy_limit)
print()

precision_coin = precision(table_coin[1][1],table_coin[1][0])
precision_limit = precision(table_limit[1][1],table_limit[1][0])

print('Precision для монетки = ', precision_coin)
print('Precision для порога = ', precision_limit)
print()

recall_coin = recall(table_coin[1][1], table_coin[0][1])
recall_limit = recall(table_limit[1][1], table_limit[0][1])

print('Recall для монетки = ', recall_coin)
print('Recall для порога = ', recall_limit)
print()

F1_Score_coin = F1_Score(precision_coin, recall_coin)
F1_Score_limit = F1_Score(precision_limit, recall_limit)

print('F1 Score для монетки = ', F1_Score_coin)
print('F1 Score для порога = ', F1_Score_limit)