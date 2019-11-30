import random
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
X = np.array([[1, 60], [1, 50], [1, 75]])
y = np.array([[10], [7], [12]])
step1 = X.T.dot(X)
step2 = np.linalg.inv(step1)
step3 = step2.dot(X.T)
b = step3.dot(y)
print(b)
