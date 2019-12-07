import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mlxtend.plotting import plot_decision_regions
from PerceptronGradient.Perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Графики 1
# setosa and versicolor
# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length
# X = df.iloc[0:100, [0, 2]].values
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

y2 = df.iloc[50:150, 4].values
y2 = np.where(y2 == 'Iris-virginica', -1, 1)

# sepal width and petal width
X2 = df.iloc[50:150, [1, 3]].values

ppn = Perceptron(epochs=25, eta=0.01)
ppn.train(X2, y2)

plot_decision_regions(X2, y2, clf=ppn)
plt.title('Perceptron')
plt.xlabel('Длина чашелистика [cm]')
plt.ylabel('Длина лепестка [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Итерации')
plt.ylabel('Ошибочная классификация')
plt.show()

print('Общее число шобочных классификаций: %d из 100' % (y2 != ppn.predict(X2)).sum())

