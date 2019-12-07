import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.plotting import plot_decision_regions
from PerceptronGradient.PerceptronGradientBoost import PerceptronGradientBoost

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Графики 1
# setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
#
# ppn = Perceptron(epochs=10, eta=0.1)
#
# ppn.train(X, y)
# print('Weights: %s' % ppn.w_)
# plot_decision_regions(X, y, clf=ppn)
# plt.title('Perceptron')
# plt.xlabel('Длина чашелистика [cm]')
# plt.ylabel('Длина лепестка [cm]')
# plt.show()
#
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Итерации')
# plt.ylabel('Ошибочная классификация')
# plt.show()


# Графики 2
# setosa and versicolor

# y2 = df.iloc[50:150, 4].values
# y2 = np.where(y2 == 'Iris-virginica', -1, 1)
#
# sepal width and petal width
# X2 = df.iloc[50:150, [1, 3]].values
#
# ppn = Perceptron(epochs=25, eta=0.01)
# ppn.train(X2, y2)
#
# plot_decision_regions(X2, y2, clf=ppn)
# plt.title('Perceptron')
# plt.xlabel('Длина чашелистика [cm]')
# plt.ylabel('Длина лепестка [cm]')
# plt.show()
#
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Итерации')
# plt.ylabel('Ошибочная классификация')
# plt.show()
#
# print('Общее число шобочных классификаций: %d из 100' % (y2 != ppn.predict(X2)).sum())


# Обучение Персептрона с градиентом без стандартизации
# pgb = PerceptronGradientBoost(epochs=10, eta=0.01).train(X, y)
# plt.plot(range(1, len(pgb.cost_) + 1), np.log10(pgb.cost_), marker='o')
# plt.xlabel('Итерации')
# plt.ylabel('log2(Сумма квадратов ошибок)')
# plt.title('Параметр learning rate = 0.01')
# plt.show()
#
# pgb = PerceptronGradientBoost(epochs=10, eta=0.0001).train(X, y)
# plt.plot(range(1, len(pgb.cost_) + 1), pgb.cost_, marker='o')
# plt.xlabel('Итерации')
# plt.ylabel('Сумма квадратов ошибок')
# plt.title('Параметр learning rate = 0.0001')
# plt.show()


# Обучение Персептрона с градиентом с стандартизацией
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

pgb = PerceptronGradientBoost(epochs=15, eta=0.01)

pgb.train(X_std, y)
plot_decision_regions(X_std, y, clf=pgb)
plt.title('Персептрон с градинтным спуском')
plt.xlabel('Длина чашелистика [стандартизованная]')
plt.ylabel('Длина лепестка [стандартизованная]')
plt.show()

plt.plot(range(1, len(pgb.cost_) + 1), pgb.cost_, marker='o')
plt.xlabel('Итерации')
plt.ylabel('Сумма квадратов ошибок')
plt.show()
