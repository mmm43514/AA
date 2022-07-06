#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:22:50 2021

@author: Mario Muñoz Mesa
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt # (parte 1)
from sklearn.model_selection import train_test_split # (parte 2)

# PARTE 1
iris = datasets.load_iris() # cargamos base datos iris

print('Veamos que hay disponible en la base de datos iris con .keys()')
print(iris.keys())
print('Veamos las primeras líneas de la descripción de la base datos iris')
for line in (iris.DESCR.splitlines())[1:18:1]:
    print(line)
print('\t...\n')
n, m = iris.data.shape
print('La base datos contiene {} elementos, cada uno con {} características o atributos'.format(n,m))

X = iris.data # datos de entrada

print('Los nombres de las características de cada elemento son')
print(iris.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

y = iris.target # clases

print('Los nombres de las clases son')
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']

c1 = X[:,0]
c2 = X[:,2]

print('Los primeros 10 datos de la primera característica son')
print(c1[0:10:1])
print('Los primeros 10 datos de la tercera característica son')
print(c2[0:10:1])
       

# Scatter Matrix
fig, ax = plt.subplots(m, m, figsize=(20, 20))

for i in range(m):
    for j in range(m):
        for clase, color in zip(np.unique(y), ('orange','black','green')):
            ax[i, j].scatter(X[y==clase, i], 
                             X[y==clase, j],
                             label = iris.target_names[clase],
                             color = color)
        ax[i, j].set_xlabel(iris.feature_names[i])
        ax[i, j].set_ylabel(iris.feature_names[j])
        ax[i, j].legend()

plt.show()

input('Pulse enter para continuar')
# PARTE 2
# stratify = y para conservar la proporción de elementos de cada clase tanto en training como en test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    train_size = 0.75, 
                                                    stratify = y)
print('Cantidad de elementos en el entrenamiento: {}'.format(len(X_train)))
print('De los cuales {} son de clase 0, {} de clase 1 y {} de clase 2'.format(len(X_train[y_train==0]), 
                                                                              len(X_train[y_train==1]), 
                                                                              len(X_train[y_train==2])))
print('Cantidad de elementos en el test: {}'.format(len(X_test)))
print('De los cuales {} son de clase 0, {} de clase 1 y {} de clase 2'.format(len(X_test[y_test==0]), 
                                                                              len(X_test[y_test==1]), 
                                                                              len(X_test[y_test==2])))


input('Pulse enter para continuar')
# PARTE 3
# Valores equiespaciados entre 0 y 4*PI
ev_spaced = np.linspace(0, 4*np.pi, 100)
print('100 valores equiespaciados entre 0 y 4*pi')
print(ev_spaced)

# Obtenemos sin(x), cos(x) y tanh(sin(x)+cos(x)) de los valores anteriores
s = np.sin(ev_spaced)
c = np.cos(ev_spaced)
th = np.tanh(s+c)
print('\nMostramos 10 primeros valores de sin(x), cos(x) y tanh(sin(x)+cos(x))')
print('sin(x):')
print(s[0:10:1])
print('cos(x):')
print(c[0:10:1])
print('tanh(sin(x)+cos(x)):')
print(th[0:10:1])

# Visualizamos las 3 líneas en verde, negro y rojo (discontinuas)
plt.plot(ev_spaced, s, 'g--', label='sin(x)')
plt.plot(ev_spaced, c, 'k--', label='cos(x)')
plt.plot(ev_spaced, th, 'r--', label='tanh(sin(x)+cos(x))')
plt.legend(loc='lower left', prop={'size': 7})
plt.title('Tres curvas en un mismo plot')
plt.show()