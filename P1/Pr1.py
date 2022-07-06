# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Mario Muñoz Mesa
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
rng = np.random.default_rng(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# Función a la que queremos encontrar mínimo
def E(u,v):
    return (u**3 * np.e**(v-2) - 2 * v**2 * np.e**(-u))**2

# Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2 * (u**3 * np.e**(v-2) - 2 * v**2 * np.e**(-u)) * (3 * u**2 * np.e**(v-2) + 2 * v**2 * np.e**(-u))
    
# Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2 * (u**3 * np.e**(v-2) - 2 * v**2 * np.e**(-u)) * (u**3 * np.e**(v-2) - 4 * np.e**(-u) * v)

# Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# EJERCICIO 1.1
# Algoritmo Gradiente Descendente (GD)
'''
   Implementación algoritmo Gradiente Descendente

   :param function funct: función a minimizar
   :param function grad_funct: gradiente de la función a minimizar
   :param numpy.ndarray w0: punto inicial
   :param learning_rate: tasa de aprendizaje
   :param max_iter: iteraciones máximas
   :param error: cota de error
   :return: el último valor iterado, iteraciones totales realizadas
'''
def gradient_descent(funct, grad_funct, w0, learning_rate,  max_iter, error):
    w = w0
    iterations = 0
    # Mientras no tengamos condición de parada por iteraciones ni alcancemos la cota de error
    while iterations < max_iter and error < funct(w[0], w[1]):
        # Actualizar punto w ''dando el paso en la dirección más profunda'' (proporcional a la tasa de aprendizaje)
        w = w - learning_rate * grad_funct(w[0], w[1]) 
        iterations += 1
        
    return w, iterations    

# EJERCICIO 1.2 (b)
print('Ejercicio 1.2 (b)')

eta = 0.1 # tasa de aprendizaje 0.1
maxIter = 10000000000
error2get = np.double(10**(-14))
initial_point = np.array([1.0, 1.0], dtype=np.double)
w, it = gradient_descent(E, gradE, initial_point, eta, maxIter, error2get)

print('Ejercicio 1.2 (c)')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')') # EJERCICIO 1.2 (c)


# DISPLAY FIGURE. Representación gráfica de E
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0], w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
del x
del y
#Seguir haciendo el ejercicio...
# Función a la que queremos encontrar mínimo
def f(x,y):
    # Devuelve f evaluada en (x,y)
    return  (x + 2)**2 + 2*(y - 2)**2  + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Derivada parcial de f con respecto a x
def dfx(x,y):
    # Devuelve la derivada parcial de f respecto a x evaluada en (x,y)
    return 2 * (2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) + x + 2)
    
# Derivada parcial de f con respecto a y
def dfy(x,y):
    # Devuelve la derivada parcial de f respecto a y evaluada en (x,y)
    return 4 * (np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + y - 2)

# Gradiente de f
def gradf(x,y):
    # Devuelve el gradiente de f evaluado en (x,y)
    return np.array([dfx(x,y), dfy(x,y)])   

# EJERCICIO 1.3 (a)
print('Ejercicio 1.3 (a)')
# Función que muestra los valores de la función, sobre la que se usa Gradiente Descendiente, tomados en cada iteración
'''
   Implementación de función que dibuja los valores que toma una función en cada iteración del método Gradiente Descendente

   :param function funct: función a minimizar con GD y dibujar valores en cada iteración
   :param function grad_funct: gradiente de la función funct
   :param numpy.ndarray w0: punto inicial para GD
   :param learning_rate: tasa de aprendizaje para GD
   :param max_iter: iteraciones máximas en GD
'''
def gradient_descent_graphic_f_values(funct, grad_funct, w0, learning_rate,  max_iter):
    w = w0
    iterations = 0
    f_values = np.array([]) # array en el que guardaremos valores de f en w
    f_values = np.append(f_values, funct(w[0], w[1])) # guardamos valor inicial
    # Mientras no tengamos condición de parada por iteraciones ni alcancemos el error objetivo
    while iterations < max_iter:
        # Actualizar punto w ''dando el paso en la dirección más profunda'' (proporcional a la tasa de aprendizaje)
        w = w - learning_rate * grad_funct(w[0], w[1]) 
        f_values = np.append(f_values, funct(w[0], w[1])) # guardamos valor de f en w
        iterations += 1
    # Hacemos plot de las iteraciones y los valores de f obtenidos en cada iteración
    plt.plot(np.arange(max_iter+1), f_values, linestyle='--', marker='o', color='k')
    plt.title('Valores de f por iteración (algoritmo GD)\n learning rate={}, punto inicial={}'.format(learning_rate, w0))
    plt.xlabel('Número iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()

print('Vamos a dibujar las gráficas de valores en f en cada iteración de Gradiente Descendente para learning rate 0.01 y 0.1')
# Usamos primero tasa de aprendizaje 0.01
eta = 0.01
maxIter = 50
initial_point = np.array([-1.0,1.0], dtype=np.double)
# Mostramos la gráfica, de valores de f por iteración, para eta=0.01 y un máximo de 50 iteraciones
print('Mostramos la gráfica, de valores de f por iteración, para eta=0.01 y un máximo de 50 iteraciones')
gradient_descent_graphic_f_values(f, gradf, initial_point, eta, maxIter)

# Probamos ahora con tasa de aprendizaje 0.1
eta = 0.1
# Mostramos la gráfica, de valores de f por iteración, para eta=0.1 y un máximo de 50 iteraciones
print('Mostramos la gráfica, de valores de f por iteración, para eta=0.1 y un máximo de 50 iteraciones')
gradient_descent_graphic_f_values(f, gradf, initial_point, eta, maxIter)

input("\n--- Pulsar tecla para continuar ---\n")

# EJERCICIO 1.3 (b)
print('Ejercicio 1.3 (b)')
# Utilizaremos tasa de aprendizaje 0.01 y un máximo de 50 iteraciones. 
# Probaremos con distintos puntos iniciales: (-0.5, -0.5), (1, 1), (2.1, -2.1), (-3, 3), (-2, 2)
print('Utilizaremos tasa de aprendizaje 0.01 y un máximo de 50 iteraciones.')
print('Probaremos con distintos puntos iniciales: (-0.5, -0.5), (1, 1), (2.1, -2.1), (-3, 3), (-2, 2)')

# Función que devuelve los valores que toma una función junto con los puntos donde los toma por cada iteración del método del Gradiente Descendente
'''
   Implementación de función que devuelve los valores que toma una función junto con el punto donde lo toma en cada iteración del método Gradiente Descendente
   También dibuja los valores de f por iteración

   :param function funct: función a minimizar con GD y dibujar valores en cada iteración
   :param function grad_funct: gradiente de la función funct
   :param numpy.ndarray w0: punto inicial para GD
   :param learning_rate: tasa de aprendizaje para GD
   :param max_iter: iteraciones máximas en GD
   :return: pesos w y valores de f(w) tomados en cada iteración
'''
def gradient_descent_get_f_and_w_values(funct, grad_funct, w0, learning_rate,  max_iter):
    w = w0
    iterations = 0
    f_values = np.array([], dtype=np.double) # array en el que guardaremos valores de f en w
    f_values = np.append(f_values, funct(w[0], w[1])) # guardamos valor de w inicial
    w_values = [] # lista que guardará los valores de w (utilizamos lista de python por comodidad y porque serán pocos datos)
    w_values.append(w0) # guardamos el punto de inicio
    # Mientras no tengamos condición de parada por iteraciones ni alcancemos el error objetivo
    while iterations < max_iter:
        # Actualizar punto w ''dando el paso en la dirección más profunda'' (proporcional a la tasa de aprendizaje)
        w = w - learning_rate * grad_funct(w[0], w[1]) 
        f_values = np.append(f_values, funct(w[0], w[1])) # guardamos valor de f en w
        w_values.append(w) # guardamos el valor de w actual
        iterations += 1
    # Hacemos plot de las iteraciones y los valores de f obtenidos en cada iteración
    plt.plot(np.arange(max_iter+1), f_values, linestyle='--', marker='o', color='k')
    plt.title('Valores de f por iteración (algoritmo GD)\n learning rate={}, punto inicial={}'.format(learning_rate, w0))
    plt.xlabel('Número iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()
    #print(f_values)
    return w_values, f_values
    
eta = 0.01 # tasa de aprendizaje 0.01
maxIter = 50
initial_points = np.array([[-0.5, -0.5],[1,1],[2.1,-2.1],[-3,3],[-2,2]], dtype=np.double)

# Ejecutamos la función gradient_descent_get_f_and_w_values para los puntos (-0.5, -0.5), (1, 1), (2.1, -2.1), (-3, 3), (-2, 2)
for i_p in initial_points:
    w_values, f_values = gradient_descent_get_f_and_w_values(f, gradf, i_p, eta, maxIter)
    print('Punto inicial ({}, {})'.format(i_p[0], i_p[1]))
    #print ('Numero de iteraciones: ', it)
    #print ('w_values ', w_values)
    #print('f_values', f_values)
    print('Último valor (x,y)', w_values[-1])
    print('Último valor f(x,y)', f_values[-1])
    

# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], f(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(X,y,w):
    M = np.double(X.shape[0])
    return (1/M)*np.linalg.norm(X.dot(w)-y)**2

# Gradiente en minibatch
def grad_Ein_minibatch(X, y, w):
    M = np.double(X.shape[0])
    return (2/M)*(X.T).dot(X.dot(w)-y)

# Gradiente Descendente Estocastico
'''
   Implementación de algoritmo Gradiente Descendente Estocástico. (condición de parada por iteraciones)
   
   :param numpy.ndarray X: matriz muestral de vectores de características
   :param numpy.ndarray y: etiquetas o clases de la muestra
   :param numpy.ndarray w0: vector de pesos inicial
   :param learning_rate: tasa de aprendizaje para GDE
   :param max_iter: iteraciones máximas en GDE
   :return: el último vector de pesos calculado
'''
def sgd(X, y, w0, learning_rate, minibatches_size, max_iterations):
    w = w0 
    sample_size = X.shape[0] # tamaño de muestra
    indexes = np.arange(sample_size) # trabajaremos con índices
    # Mezclamos aleatoriamente la muestra (trabajando con índices) por primera vez
    np.random.shuffle(indexes)
    # Marcamos el primer minibatch
    minibatch_init = 0
    minibatch_end = minibatch_init + minibatches_size
    
    # Hasta que no alcancemos el máximo de iteraciones
    for i in range(max_iterations):
        # Si ya hemos iterado en todos los minibatches
        if (minibatch_init > sample_size):
            # Mezclamos aleatoriamente muestra (trabajando con índices), generando así nuevos minibatches
            np.random.shuffle(indexes)
            # reestablecemos valores para empezar por primer minibatch
            minibatch_init = 0
            minibatch_end = minibatch_init + minibatches_size
        
        # Se toma el minibatch actual
        minibatch = indexes[minibatch_init:minibatch_end]
        # Se calcula el vector pesos para el minibatch tomado (dando el 'paso de mayor profundidad' en el minibatch)
        w = w - learning_rate * grad_Ein_minibatch(X[minibatch], y[minibatch], w)
        # Se avanza al siguiente minibatch
        minibatch_init += minibatches_size
        minibatch_end += minibatches_size
        
    return w

# Pseudoinversa	
'''
   Implementación de método por pseudoinversa.
   
   :param numpy.ndarray X: matriz muestral de vectores de características
   :param numpy.ndarray y: etiquetas o clases de la muestra
   :return: el vector de pesos que minimiza el ECM para la muestra
'''
def pseudoinverse(X, y):
    # Descomposición por valores singulares de X
    U, d, Vt = np.linalg.svd(X) 
    # Construimos la matriz D_, matriz que en su diagonal tiene 1 entre el cuadrado de cada valor singular o 0 si el valor singular es 0
    d = np.array([np.inf if i == 0 else i for i in d]) # cambiamos los ceros por np.inf (para dividir después y obtener D_)
    d = 1 / (d*d)
    D_ = np.diag(d)
    # Devolvemos el producto de la pseudoinversa por y, (es decir el vector de pesos que minimiza el ECM para la muestra)
    return ( ( ( (Vt.T).dot(D_) ).dot(Vt) ).dot(X.T) ).dot(y)


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(x, y, np.zeros(x.shape[1]), 0.01, 32, 1000)
print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

print ('Vector de pesos obtenido: ', w)

# Dibujamos Intensidad promedio y Simetría de dígitos manuscritos 1 y 5 junto con la recta de regresión obtenido por SGD
ev_spaced = np.linspace(0.05, 0.55, 500)

plt.plot(ev_spaced, -(w[0]+w[1] * ev_spaced) / w[2], 'k-', label='Recta de regresión SGD')
plt.plot(x[y==-1, 1], x[y==-1, 2], 'r.', label = '1')
plt.plot(x[y==1, 1], x[y==1, 2], 'b.', label = '5')
plt.title('Regresión por SGD para clasificar dígitos manuscritos 1 y 5\n en base a intensidad promedio y simetría')
plt.legend(loc='upper right', prop={'size': 7})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...
w_lin = pseudoinverse(x,y)
print ('Bondad del resultado para método pseudoinversa:\n')
print ("Ein: ", Err(x,y,w_lin))
print ("Eout: ", Err(x_test, y_test, w_lin))

print ('Vector de pesos obtenido: ', w_lin)

# Dibujamos Intensidad promedio y Simetría de dígitos manuscritos 1 y 5 junto con la recta de regresión obtenido por pseudoinverse
ev_spaced = np.linspace(0.05, 0.55, 500)

plt.plot(ev_spaced, -(w_lin[0]+w_lin[1] * ev_spaced) / w_lin[2], 'k-', label='Recta de regresión Pseudoinversa')
plt.plot(x[y==-1, 1], x[y==-1, 2], 'r.', label = '1')
plt.plot(x[y==1, 1], x[y==1, 2], 'b.', label = '5')
plt.title('Regresión por Pseudoinversa para clasificar dígitos manuscritos 1 y 5\n en base a intensidad promedio y simetría')
plt.legend(loc='upper right', prop={'size': 7})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.show()

# Dibujamos ambas rectas de regresión en mismo plot
ev_spaced = np.linspace(0.05, 0.55, 500)
plt.plot(ev_spaced, -(w[0]+w[1] * ev_spaced) / w[2], 'k-', label='Recta de regresión SGD')
plt.plot(ev_spaced, -(w_lin[0]+w_lin[1] * ev_spaced) / w_lin[2], 'g-', label='Recta de regresión Pseudoinversa')
plt.plot(x[y==-1, 1], x[y==-1, 2], 'r.', label = '1')
plt.plot(x[y==1, 1], x[y==1, 2], 'b.', label = '5')
plt.title('Regresión por Pseudoinversa y SGD para clasificar dígitos manuscritos\n 1 y 5 en base a intensidad promedio y simetría')
plt.legend(loc='upper right', prop={'size': 7})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d)) # (N,d) indica N puntos de d-dimensionales

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1 - 0.2)**2 + x2**2 - 0.6) 

#Seguir haciendo el ejercicio...


# Ejercicio 2.2 (a)
print('Ejercicio 2.2 (a)')
# Generamos los 1000 puntos bidimensionales siguiendo una distribución uniforme en [-1,1]
unif_points_2d = simula_unif(1000, 2, 1)
# Dibujamos los puntos anteriores
plt.plot(unif_points_2d[:,0],unif_points_2d[:,1], 'k.')
plt.title('Mapa de 1000 puntos 2-dimensionales de distribución uniforme en [-1,1]')
plt.xlabel('Primera coordenada del punto')
plt.ylabel('Segunda coordenada del punto')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 2.2 (b)
print('Ejercicio 2.2 (b)')
# Tomamos 100 puntos (10% de 1000) aleatorios para utilizarlos como puntos con ruido en las etiquetas

noise_points = rng.choice(1000, size=100, replace=False)
#noise_points = np.random.randint(0, 1000, 100)
f_vct = np.vectorize(f) # vectorizamos la función para poder pasar arrays como argumentos
target = f_vct(unif_points_2d[:,0], unif_points_2d[:,1]) # generamos las etiquetas mediante f
# aplicamos ruido en las etiquetas de los puntos noise_points (elegidos al azar anteriormente)
target[noise_points] = -target[noise_points] 

# Dibujamos los puntos anteriores con sus clases
for label, color in zip([-1,1], ('red', 'blue')):
    plt.scatter(unif_points_2d[target==label, 0],
                unif_points_2d[target==label, 1],
                label=label,
                color=color,
                marker='.')
plt.title('1000 puntos 2-dimensionales siguiendo distrib. uniforme, clasificados\n mediante f y con ruido en 10% de las etiquetas')
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 2.2 (c)
print('Ejercicio 2.2 (c)')
# Generamos la matriz X con las características de la muestra
X = np.array([[1, unif[0], unif[1]] for unif in unif_points_2d])
y = target

# Modelo regresión por pseudoinversa
w_lin = pseudoinverse(X,y)
print ('Bondad del resultado para método pseudoinversa:\n')
print ("Ein: ", Err(X,y,w_lin))
print ('Vector de pesos obtenido: ', w_lin)

# Modelo regresión por SGD
w = sgd(X, y, np.zeros(X.shape[1]), 0.01, 32, 1000)
print ('\nBondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(X,y,w))
print ('Vector de pesos obtenido: ', w)

# Dibujamos ambas rectas de regresión en mismo plot
ev_spaced = np.linspace(-1, 1, 1000)
plt.plot(ev_spaced, -(w[0]+w[1] * ev_spaced) / w[2], 'k-', label='Recta de regresión SGD')
plt.plot(ev_spaced, -(w_lin[0]+w_lin[1] * ev_spaced) / w_lin[2], 'g-', label='Recta de regresión Pseudoinversa')
plt.plot(X[y==-1, 1], X[y==-1, 2], 'r.', label = '-1')
plt.plot(X[y==1, 1], X[y==1, 2], 'b.', label = '1')
plt.ylim(-1.1, 1.1)
plt.title('Regresión por Pseudoinversa y SGD ')
plt.legend(loc='upper right', prop={'size': 6})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 2.2 (d)
print('Ejercicio 2.2 (d)')
print('...tarda en ejecutar...')
e_in_wlin_acum = 0
e_in_wsgd_acum = 0
e_out_wlin_acum = 0
e_out_wsgd_acum = 0

# bucle para repetir 1000 veces lo pedido en ej 2.2 (d)
for i in range(1000):
    # Generamos la muestra con ruido en 10% de las etiquetas
    unif_points_2d = simula_unif(1000, 2, 1)
    noise_points = rng.choice(1000, size=100, replace=False)
    
    target = f_vct(unif_points_2d[:,0], unif_points_2d[:,1])
    target[noise_points] = -target[noise_points] 
    # Generamos matriz con vector características (1, x0, x1)
    X = np.array([[1, unif[0], unif[1]] for unif in unif_points_2d])
    # Llamamos y al vector con las etiquetas de cada elemento de la muestra con 10% de ruido
    y = target
    
    w_lin = pseudoinverse(X,y) # vector de pesos obtenido por pseudoinversa
    w_sgd = sgd(X, y, np.zeros(X.shape[1]), 0.01, 32, 1000)  # vector de pesos obtenido por SGD
    #Acumulamos el error dentro de la muestra
    e_in_wlin_acum += Err(X, y, w_lin)
    e_in_wsgd_acum += Err(X, y, w_sgd)

    # Generamos la muestra para test, con ruido en 10% de las etiquetas
    test = simula_unif(1000, 2, 1)
    # Generamos las etiquetas con ruido
    noise_points_test = rng.choice(1000, size=100, replace=False)
    target_test = f_vct(test[:,0], test[:,1])
    target_test[noise_points_test] = -target_test[noise_points_test]
    # Generamos matriz del test con vector características (1, x0, x1) 
    X_test = np.array([[1, t[0], t[1]] for t in test])
    # Llamamos y a las etiquetas del test con 10% ruido ya generadas
    y_test = target_test
    # Acumulamos el error fuera de la muestra
    e_out_wlin_acum += Err(X_test, y_test, w_lin)
    e_out_wsgd_acum += Err(X_test, y_test, w_sgd)
    
print('Error medio, por pseudoinversa, dentro de la muestra', e_in_wlin_acum/1000)
print('Error medio, por SGD, dentro de la muestra', e_in_wsgd_acum/1000)

print('Error medio, por pseudoinversa, fuera de la muestra', e_out_wlin_acum/1000)
print('Error medio, por SGD, fuera de la muestra', e_out_wsgd_acum/1000)

input("\n--- Pulsar tecla para continuar ---\n")

# Repetimos para el vector de características (1, x1, x2, x1x2, x1^2x2^2)
print('Repetimos para el vector de características (1, x1, x2, x1x2, x1^2x2^2)')

# Ejercicio 2.2 (a)
unif_points_2d = simula_unif(1000, 2, 1)
plt.plot(unif_points_2d[:,0],unif_points_2d[:,1], 'k.')
plt.title('Mapa de 1000 puntos 2-dimensionales de distribución uniforme en [-1,1]')
plt.xlabel('Primera coordenada del punto')
plt.ylabel('Segunda coordenada del punto')
plt.show()

# Ejercicio 2.2 (b)
print('Ejercicio 2.2 (b)')
# Tomamos 100 puntos (10% de 1000) aleatorios para utilizarlos como puntos con ruido en las etiquetas
noise_points = rng.choice(1000, size=100, replace=False) #  replace=False -> sin elementos repetidos
f_vct = np.vectorize(f) # vectorizamos la función para poder pasar arrays como argumentos
target = f_vct(unif_points_2d[:,0], unif_points_2d[:,1]) # generamos las etiquetas mediante f
# aplicamos ruido en las etiquetas de los puntos noise_points (elegidos al azar anteriormente)
target[noise_points] = -target[noise_points] 

# Dibujamos los puntos anteriores con sus clases
for label, color in zip([-1,1], ('red', 'blue')):
    plt.scatter(unif_points_2d[target==label, 0],
                unif_points_2d[target==label, 1],
                label=label,
                color=color,
                marker='.')
plt.title('1000 puntos 2-dimensionales siguiendo distrib. uniforme, clasificados\n mediante f y con ruido en 10% de las etiquetas')
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.legend()
plt.show()

# Ejercicio 2.2 (c)
print('Ejercicio 2.2 (c)')
# Generamos la matriz con los nuevos vectores de características (1, x1, x2, x1*x2, x1**2, x2**2)
X = np.array([[1, unif[0], unif[1], unif[0] * unif[1], unif[0]**2, unif[1]**2] for unif in unif_points_2d])
y = target

# Modelo regresión por pseudoinversa
w_lin = pseudoinverse(X,y)
print ('Bondad del resultado para método pseudoinversa:\n')
print ("Ein: ", Err(X,y,w_lin))
print ('Vector de pesos obtenido: ', w_lin)

# Modelo regresión por SGD
w_sgd = sgd(X, y, np.zeros(X.shape[1]), 0.01, 32, 1000)
print ('\nBondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(X,y,w_sgd))
print ('Vector de pesos obtenido: ', w_sgd)

# Dibujamos regresión mediante método pseudoinversa
ev_spaced = np.linspace(-1, 1, 2000)
# extraemos primera y segunda coordenadas, de cada punto del producto cartesiano ev_spaced x ev_spaced (rejilla), en dos matrices de coordenadas
# en x1_grid las filas tienen como elementos los valores de ev_spaced. 
# En x2_grid tienen los valores de eje coord y para cada punto. La matriz queda con los valores de las filas de x1_grid pero como columnas
x1_grid, x2_grid = np.meshgrid(ev_spaced, ev_spaced)
# Para dibujar la función recurrimos a plt.contour que dibuja líneas de contorno. Indicamos la línea contorno [0] que es la de nuestra fórmula
cont = plt.contour(x1_grid, x2_grid, w_lin[0] + w_lin[1]*x1_grid + w_lin[2]*x2_grid + w_lin[3]*x1_grid*x2_grid + w_lin[4]*x1_grid**2 + w_lin[4]*x2_grid**2, [0])
cont.collections[0].set_label('Función de regresión') # ponemos etiqueta para que aparezca en la leyenda

plt.plot(X[y==-1, 1], X[y==-1, 2], 'r.', label = '-1')
plt.plot(X[y==1, 1], X[y==1, 2], 'b.', label = '1')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1, 1.1)
plt.title('Regresión por Pseudoinversa\n (1, x1, x2, x1*x2, x1**2, x2**2) ')
plt.legend(loc='upper right', prop={'size': 7})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()

# Dibujamos regresión mediante método SGD
ev_spaced = np.linspace(-1, 1, 2000)
# extraemos primera y segunda coordenadas, de cada punto del producto cartesiano ev_spaced x ev_spaced (rejilla), en dos matrices de coordenadas
# en x1_grid las filas tienen como elementos los valores de ev_spaced. 
# En x2_grid tienen los valores de eje coord y para cada punto. La matriz queda con los valores de las filas de x1_grid pero como columnas
x1_grid, x2_grid = np.meshgrid(ev_spaced, ev_spaced)
# Para dibujar la función recurrimos a plt.contour que dibuja líneas de contorno. Indicamos la línea contorno [0] que es la de nuestra fórmula
cont = plt.contour(x1_grid, x2_grid, w_sgd[0] + w_sgd[1]*x1_grid + w_sgd[2]*x2_grid + w_sgd[3]*x1_grid*x2_grid + w_sgd[4]*x1_grid**2 + w_sgd[4]*x2_grid**2, [0])
cont.collections[0].set_label('Función de regresión')

plt.plot(X[y==-1, 1], X[y==-1, 2], 'r.', label = '-1')
plt.plot(X[y==1, 1], X[y==1, 2], 'b.', label = '1')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1, 1.1)
plt.title('Regresión por SGD\n (1, x1, x2, x1*x2, x1**2, x2**2) ')
plt.legend(loc='upper right', prop={'size': 6})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")
# Ejercicio 2.2 (d)
#print('Ejercicio 2.2 (d)')
print('...tarda en ejecutar...')
e_in_wlin_acum = 0
e_in_wsgd_acum = 0
e_out_wlin_acum = 0
e_out_wsgd_acum = 0

# bucle para repetir 1000 veces lo pedido en ej 2.2 (d) (vector de caract (1, x1, x2, x1*x2, x1**2, x2**2) )
for i in range(1000):
    # Generamos la muestra con ruido en 10% de las etiquetas
    unif_points_2d = simula_unif(1000, 2, 1)
    # Generamos las etiquetas con ruido
    noise_points = rng.choice(1000, size=100, replace=False)
    target = f_vct(unif_points_2d[:,0], unif_points_2d[:,1])
    target[noise_points] = -target[noise_points] 
    # Generamos la matriz con vector características (1, x1, x2, x1*x2, x1**2, x2**2)
    X = np.array([[1, unif[0], unif[1], unif[0] * unif[1], unif[0]**2, unif[1]**2] for unif in unif_points_2d])
    # Llamamos y al vector con las etiquetas de cada elemento de la muestra con 10% de ruido
    y = target
    
    w_lin = pseudoinverse(X,y) # vector de pesos obtenido por pseudoinversa
    w_sgd = sgd(X, y, np.zeros(X.shape[1]), 0.01, 32, 1000) # vector de pesos obtenido por SGD
    #Acumulamos el error dentro de la muestra
    e_in_wlin_acum += Err(X, y, w_lin)
    e_in_wsgd_acum += Err(X, y, w_sgd)

    # Generamos la muestra para test, con ruido en 10% de las etiquetas
    test = simula_unif(1000, 2, 1)
    # Generamos las etiquetas con ruido
    noise_points_test = rng.choice(1000, size=100, replace=False)
    target_test = f_vct(test[:,0], test[:,1])
    target_test[noise_points_test] = -target_test[noise_points_test]
    # Generamos la matriz del test con vector características (1, x1, x2, x1*x2, x1**2, x2**2)
    X_test = np.array([[1, unif[0], unif[1], unif[0] * unif[1], unif[0]**2, unif[1]**2] for unif in test])
    # Llamamos y a las etiquetas del test con 10% ruido ya generadas
    y_test = target_test
    #Acumulamos el error fuera de la muestra
    e_out_wlin_acum += Err(X_test, y_test, w_lin)
    e_out_wsgd_acum += Err(X_test, y_test, w_sgd)
    
print('Error medio, por pseudoinversa, dentro de la muestra', e_in_wlin_acum/1000)
print('Error medio, por SGD, dentro de la muestra', e_in_wsgd_acum/1000)

print('Error medio, por pseudoinversa, fuera de la muestra', e_out_wlin_acum/1000)
print('Error medio, por SGD, fuera de la muestra', e_out_wsgd_acum/1000)


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
# BONUS EJERCICIO NEWTON
print('BONUS\n')
print('Ejercicio 2.1 (bonus)\n')

# Función a la que queremos encontrar mínimo . (la redefinimos para no usar f del ejercicio 2)
def f(x,y):
    # Devuelve f evaluada en (x,y)
    return  (x + 2)**2 + 2*(y - 2)**2  + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) 

# El gradiente de f ya está definido en el ejercicio 1

# Matriz Hessiana de f
def Hessiana_f(x,y):
    # Devuelve la matriz Hessiana de f evaluada en (x,y)
    return np.array(
    [[2 - 8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y), 8 * np.pi**2 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)],
     [8 * np.pi**2 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y), 4 - 8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)]]
    )
'''
   Método de Newton.
   
   :param np.array gradf: gradiente de la función a minimizar
   :param np.array Hessiana_f: matriz hessiana de la función a minimizar
   :param np.array initial_point: punto inicial, (se supone cerca de un mínimo)
   :param int max_iter: iteraciones del algoritmo
   :return: el punto del mínimo aproximado tras max_iter iteraciones
'''
def met_newton_f(gradf, Hessiana_f, initial_point, max_iter):
    w_actual = initial_point
    # Repetimos max_iter veces
    for i in range(max_iter):
        # Calculamos la inversa de la Hessiana de f evaluada en w_actual
        Hessiana_inv = np.linalg.inv(Hessiana_f(w_actual[0], w_actual[1]))
        # El nuevo punto es el actual - el producto de la Hessiana de f evaluada en w_actual por el gradiente de f evaluado en w_actual
        w_actual = w_actual - Hessiana_inv.dot(gradf(w_actual[0], w_actual[1]))
        
    return w_actual
'''
   Método de Newton. 
   Devuelve pesos tomados w, valores de f(w) en cada iteración. Dibuja esto mismo también.
   
   :param function f: función a minimizar
   :param np.array gradf: gradiente de la función a minimizar
   :param np.array Hessiana_f: matriz hessiana de la función a minimizar
   :param np.array initial_point: punto inicial, (se supone cerca de un mínimo)
   :param int max_iter: máximas iteraciones del algoritmo
   :return: pesos w y valores de f(w) tomados en cada iteración
'''
def met_newton_f_get_f_and_w_values(f, gradf, Hessiana_f, initial_point, max_iter):
    w_actual = initial_point
    f_values = np.array([], dtype=np.double) # array en el que guardaremos valores de f en w
    f_values = np.append(f_values, f(w_actual[0], w_actual[1])) # guardamos valor de f en w inicial
    w_values = [] # lista que guardará los valores de w (utilizamos lista de python por comodidad y porque serán pocos datos)
    w_values.append(w_actual)
    iterations = 0
    
    # Repetimos max_iter veces
    while iterations < max_iter:
         # Calculamos la inversa de la Hessiana de f evaluada en w_actual
        Hessiana_inv = np.linalg.inv(Hessiana_f(w_actual[0], w_actual[1]))
        # El nuevo punto es el actual - el producto de la Hessiana de f evaluada en w_actual por el gradiente de f evaluado en w_actual
        w_actual = w_actual - Hessiana_inv.dot(gradf(w_actual[0], w_actual[1]))
        f_values = np.append(f_values, f(w_actual[0], w_actual[1])) # guardamos valor de f en w actual
        w_values.append(w_actual) # guardamos el valor de w actual
        iterations += 1
        
    # Dibujamos valores de f en cada w_actual de cada iteración
    plt.plot(np.arange(max_iter+1), f_values, linestyle='--', marker='o', color='k')
    plt.title('Valores de f por iteración (método Newton)\n  punto inicial={}'.format(initial_point))
    plt.xlabel('Número iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()
        
    return w_values, f_values

# Realizamos los mismos experimentos con los mismos puntos iniciales
maxIter = 50

initial_points = np.array([[-0.5, -0.5],[1,1],[2.1,-2.1],[-3,3],[-2,2]], dtype=np.double)
# Ejecutamos la función gradient_descent_get_f_and_w_values para los puntos (-0.5, -0.5), (1, 1), (2.1, -2.1), (-3, 3), (-2, 2)
for i_p in initial_points:
    w_values, f_values = met_newton_f_get_f_and_w_values(f, gradf, Hessiana_f, i_p, maxIter)
    print('Punto inicial ({}, {})'.format(i_p[0], i_p[1]))
    print('Último valor (x,y)', w_values[-1])
    print('Último valor f(x,y)', f_values[-1])

input("\n--- Pulsar tecla para continuar ---\n")

# Ejemplo en el que sí funciona bien Método Newton. Inspirado en mínimo local de Figura 8 de la memoria
print('Ejemplo en el que sí funciona bien el Método de Newton. Inspirado en mínimo local cercano al punto de silla (Figura 8 de la memoria)')
initial_point = np.array([-1.87, 1.78], dtype=np.double) # punto inicial (-1.87, 1.78)
max_iters = 15 # iteraciones máximas
w_values, f_values = met_newton_f_get_f_and_w_values(f, gradf, Hessiana_f, initial_point, max_iters)
print('Último (x,y) obtenido', w_values[-1])
print('Último f(x,y) obtenido', f_values[-1])


input("\n--- Pulsar tecla para continuar ---\n")

print("Ejemplo en el que también funciona el método Newton. Inspirado en el ejercicio 2 apartado 1 (regresión dígitos manuscritos)")

# Volvemos a cargar los datos (variable y fue sobrescrita)
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')

# Gradiente  Ein
def grad_Ein(X, y, w):
    M = np.double(X.shape[0])
    return (2/M)*(X.T).dot(X.dot(w)-y)

'''
   Método de Newton.
   Devuelve pesos tomados w, valores de Xw en cada iteración. Dibuja esto mismo también.
   
   :param np.array x: matriz muestral de vectores de características
   :param np.array y: target (etiquetas asociadas a la muestra)
   :param function grad_Ein: función gradiente de Ein
   :param np.array Hessiana_ein: matriz hessiana de la función a minimizar, Ein
   :param np.array initial_point: punto inicial, (se supone cerca de un mínimo)
   :param int max_iter: máximas iteraciones del algoritmo
   :return: pesos w y valores de Xw tomados en cada iteración
'''
def met_newton_X_get_Ein_and_w_values(x, y, grad_Ein, Hessiana_ein, initial_point, max_iter):
    w_actual = initial_point
    ein_values = np.array([], dtype=np.double) # array en el que guardaremos valores de f en w
    ein_values = np.append(ein_values, Err(x, y, w_actual)) # guardamos valor de Xw en w inicial
    w_values = [] # lista que guardará los valores de w (utilizamos lista de python por comodidad y porque serán pocos datos)
    w_values.append(w_actual)
    iterations = 0
    
    # Repetimos max_iter veces
    while iterations < max_iter:
         # Calculamos la inversa de la Hessiana de f evaluada en w_actual
        Hessiana_inv = np.linalg.inv(Hessiana_ein)
        # El nuevo punto es el actual - el producto de la Hessiana de f evaluada en w_actual por el gradiente de f evaluado en w_actual
        w_actual = w_actual - Hessiana_inv.dot(grad_Ein(x, y, w_actual))
        ein_values = np.append(ein_values, Err(x, y, w_actual)) # guardamos valor de f en w actual
        w_values.append(w_actual) # guardamos el valor de w actual
        iterations += 1
        
    # Dibujamos valores de f en cada w_actual de cada iteración
    plt.plot(np.arange(max_iter+1), ein_values, linestyle='--', marker='o', color='k')
    plt.title('Valores de Ein por iteración (método Newton)\n (regresión dígitos manuscritos)\n  punto inicial={}'.format(initial_point))
    plt.xlabel('Número iteraciones')
    plt.ylabel('Ein')
    plt.show()
        
    return w_values, ein_values

initial_point = np.array([-1.24069346, -0.17997056, -0.46016084], dtype=np.double) # punto inicial (-1.24069346, -0.17997056, -0.46016084)
max_iters = 15 # iteraciones máximas
# Necesitaremos la Hessiana de Ein  Hess(E_in) = (2/x.shape[0])*(x.T).dot(x)
hess = (2/x.shape[0])*(x.T).dot(x)
w_values, ein_values = met_newton_X_get_Ein_and_w_values(x, y, grad_Ein, (2/x.shape[0])*(x.T).dot(x), initial_point, max_iters)
print('Último w obtenido', w_values[-1])
print('Último Ein obtenido', ein_values[-1])

