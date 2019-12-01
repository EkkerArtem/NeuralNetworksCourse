import random
import urllib
from urllib.request import urlopen

import numpy as np

# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a)
# print(a.shape)

# print(np.eye(5, 3, k=-1))

# w = np.array(random.sample(range(1000), 12))  # одномерный массив из 12 случайных чисел от 1 до 1000
# w = w.reshape((2, 2, 3))  # превратим w в трёхмерную матрицу
# print(w)

# print(w.transpose(0, 2, 1))

# ww = np.array(random.sample(range(1000), 12))  # одномерный массив из 12 случайных чисел от 1 до 1000
# print(np.asmatrix(ww).T)

# v = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
# print(v)
# print(v.mean(axis=0))  # вдоль столбцов
# print(v.mean(axis=1))  # вдоль строк
# print(v.mean(axis=None))  # вдоль всего массива

# w = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# a = w.dot([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# print(a)

# f = urlopen("https://stepic.org/media/attachments/lesson/16462/boston_houses.csv")
# sbux = np.loadtxt(f, skiprows=1, delimiter=",", unpack=True)
# print(sbux.mean(axis=1))

# arr = ((0, -0.5), (1, 0), (0, 0.5), (0, 1.5), (1, 2.5), (2,3))
# def sqerr(a, b):
#   return (a - b)**2
# def abserr(a, b):
#   return abs(a - b)
# def errsum(err_fun):
#   return sum(list(map(lambda pair: err_fun(pair[0], pair[1]), arr)))
# print(str(max(errsum(abserr), errsum(sqerr))))


# Оценка коэфициетов линейной регрессии для
# ⎛⎝D  |  V
#   10  | 60
#    7  | 50
#   12⎠ | 75
# где D- тормозной путь автомобиля, V- скорость автомобил, результаты - b0 и b1 (b - бета с крышечкой)
# X = np.array([[1, 60], [1, 50], [1, 75]])
# y = np.array([[10], [7], [12]])
# step1 = X.T.dot(X)
# step2 = np.linalg.inv(step1)
# step3 = step2.dot(X.T)
# b = step3.dot(y)
# print(b)


# f = urllib.request.urlopen("https://stepic.org/media/attachments/lesson/16462/boston_houses.csv")  # open file from URL
# data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
#
# ones = np.ones_like(data[:, :1])
# X = np.hstack((ones, data[:, 1:]))
# y = data[:, :1]
# step1 = X.T.dot(X)
# step2 = np.linalg.inv(step1)
# step3 = step2.dot(X.T)
# b = step3.dot(y)
# for i in b:
#     print(float(i), end=" ")



#Перцептрон
# Задача - https://stepik.org/lesson/21775/step/3?unit=5191
# X = np.array([[1, 1, 0.3, 1],
#               [1, 0.4, 0.5, 1],
#               [1, 0.7, 0.8, 0]])  # массив значений
#
# w = np.array([0, 0, 0])  # массив весов
#
# Y = np.array([1, 1, 0])  # реальные ответы
# X = np.array([[1, 1, 0.3, 0],
#               [1, 0.4, 0.5, 0],
#               [1, 0.7, 0.8, 0]])  # массив значений
# X[:, 3] = Y.transpose()  # подтягиваем ответы в массив значений
# w = np.array([0, 0, 0])  # массив весов
#
#
# def Predict(X):
#     E = X.dot(w.transpose())  # сумматор
#     p = 1 if E > 0 else 0  # активатор
#     return p
#
#
# perfect = False
# while perfect == False:
#     perfect = True
#     for ex in X:
#         сравниваем предсказание с реальным ответом
        # if Predict(ex[0:3]) != ex[3]:
        #     perfect = False
        #     если ответ не совпадает, правим веса
            # w = w + (ex[3] - Predict(ex[0:3])) * ex[0:3]
# print(w)
