# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Mario Muñoz Mesa
"""
import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)
rng = np.random.default_rng(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# Definimos función que devuelve accuracy, se utilizará a partir en el apartado 2 de Ej1
'''
   Función que devuelve la medida accuracy para una estimación de y

   :param numpy.ndarray y: verdadero vector de etiquetas 
   :param numpy.ndarray y_est: estimación de vector de etiquetas
   :return: accuracy
'''
def accuracy(y, y_est):
    return (y == y_est).mean()

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('EJERCICIO 1.1')
# Suponemos vector aleatorio x=(x_1,x_2) con x_1 ~ U(-50,50), x_2 ~ U(-50,50)
# Generamos realización muestral x_{1},...,x_{50}
x = simula_unif(50, 2, [-50,50])
# Mostramos gráfica de la muestra generada
plt.scatter(x[:,0], x[:,1])
plt.title('Realización muestral de tamaño 50 de vector aleatorio bidimensional\n con dist. uniforme en [-50, 50] en cada componente')
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()

# Suponemos vector aleatorio x=(x_1,x_2) con x_1 ~ N(0,5), x_2 ~ N(0,7)
# Generamos realización muestral x_{1},...,x_{50}
x = simula_gaus(50, 2, np.array([5,7]))
# Mostramos gráfica de la muestra generada
plt.scatter(x[:,0], x[:,1])
plt.title('Realización muestral de tamaño 50 de vector aleatorio bidimensional\n con dist. N(0,5) 1a componente y dist. N(0,7) en 2a componente')
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('EJERCICIO 1.2 apartado a)')
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

# Suponemos vector aleatorio x=(x_1,x_2) con x_1 ~ U(-50,50), x_2 ~ U(-50,50)
# Generamos realización muestral x_{1},...,x_{100}
x = simula_unif(100, 2, [-50, 50])
# Simulamos recta en [-50, 50] (obtenemos coeficientes de la recta)
a, b = simula_recta([-50, 50])
print('Para recta y=ax+b obtenemos: a={}, b={}'.format(a,b))
f_vct = np.vectorize(f) # vectorizamos la función f para poder pasar arrays como argumentos
y = f_vct(x[:,0], x[:,1], a, b) # obtenemos las etiquetas generadas por la función f
true_y = y.copy() # Guardamos copia del etiquetado, se utilizará en apartados posteriores

# Generamos gráfica con la muestra etiquetada y la recta utilizada para etiquetar
ptos_eval_recta = np.array([-50.0, 50.0], dtype=np.double) # guardamos valores [-50, 50] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta usada por f para clasificar') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[y==1, 0], x[y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Realización muestral de tamaño 50 de vector aleatorio bidimensional\n con dist. uniforme [-50, 50] en cada componente, clasificada mediante f')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()

# Vemos el número de puntos que tienen etiqueta -1 y +1 respectivamente
num_ptos_neg = x[y==-1].shape[0] # número de puntos muestrales con etiqueta -1
print('Número de puntos con etiqueta -1: {}'.format(num_ptos_neg))
num_ptos_pos = x[y==1].shape[0] # número de puntos muestrales con etiqueta +1
print('Número de puntos con etiqueta +1: {}'.format(num_ptos_pos))

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido
print('EJERCICIO 1.2 apartado b)')

# obtenemos vector de índices aleatorios de tamaño: 10% de la cantidad de puntos con etiqueta -1
noise_points_neg_l = rng.choice(num_ptos_neg, size=round(num_ptos_neg*0.1), replace=False) # replace=False, sin repetidos
# obtenemos vector de índices aleatorios de tamaño: 10% de la cantidad de puntos con etiqueta +1
noise_points_pos_l = rng.choice(num_ptos_pos, size=round(num_ptos_pos*0.1), replace=False) # replace=False, sin repetidos
y[ np.where(y==-1)[0][noise_points_neg_l] ] *= -1 # cambiamos la etiqueta del 10% de índices aleatorios de puntos con etiqueta -1 a +1
y[ np.where(y==1)[0][noise_points_pos_l] ] *= -1 # cambiamos la etiqueta del 10% de índices aleatorios de puntos con etiqueta +1 a -1

ptos_eval_recta = np.array([-50.0, 50.0], dtype=np.double) # guardamos valores [-50, 50] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta usada por f para clasificar') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[y==1, 0], x[y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Realización muestral de tamaño 100 de vector aleatorio bidimensional con dist. uniforme [-50, 50]\n en cada componente, clasificada mediante f y con ruido del 10% en cada etiqueta')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.show()

print('Hemos cambiado de etiqueta un {:3.4f} % de los puntos que tenían etiqueta +1.'.format((noise_points_pos_l.size / num_ptos_pos) * 100))
#print('Número de falsos positivos: {}, de un total de {} positivos. '.format(noise_points_pos_l.size, num_ptos_pos))
#print('Visto desde la clasificación que nos da f estamos obteniendo {} falsos positivos')
print('Hemos cambiado de etiqueta un {:3.4f} % de los puntos que tenían etiqueta -1.'.format((noise_points_neg_l.size / num_ptos_neg) * 100))
#print('Número de falsos negativos: {}, de un total de {} negativos.'.format(noise_points_neg_l.size, num_ptos_neg))
print('Accuracy = {}'.format(accuracy(true_y, y)))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta
print('EJERCICIO 1.2 apartado c)')
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
#CODIGO DEL ESTUDIANTE

def f1_plt(x):
    return (x[:,0]-10)**2+(x[:,1]-20)**2-400
def f2_plt(x):
    return 0.5*(x[:,0]+10)**2+(x[:,1]-20)**2-400
def f3_plt(x):
    return 0.5*(x[:,0]-10)**2-(x[:,1]+20)**2-400
def f4_plt(x):
    return x[:,1]-20*x[:,0]**2-5*x[:,0]+3

def f1(x, y):
    return signo( (x-10)**2 + (y-20)**2 - 400 )
def f2(x, y):
    return signo( 0.5*(x+10)**2 + (y-20)**2 - 400 )
def f3(x, y):
    return signo( 0.5*(x-10)**2 - (y+20)**2 - 400 )
def f4(x, y):
    return signo( y - 20*x**2 - 5*x + 3 )

# vectorizamos las funciones f1, f2, f3 y f4 para poder pasarles arrays como argumentos
f1_vct = np.vectorize(f1)
f2_vct = np.vectorize(f2)
f3_vct = np.vectorize(f3)
f4_vct = np.vectorize(f4)

# Visualizaremos muestra, de tamaño 100, de dist Unif en [-50, 50] clasificados por recta con 10% ruido en cada etiqueta, mediante la clasificación que daría f1, f2, f3 y f4
print('Visualizaremos muestra, de tamaño 100, de dist Unif en [-50, 50] clasificados por recta con 10% ruido en cada etiqueta, mediante la clasificación que darían distintas f')

#title = 'Muestra, de tamaño 100, de dist. Uniforme en [-50, 50],\n clasificados mediante f. Con ruido del 10% en cada etiqueta'
#title = 'Visualización de muestra, de tamaño 100, de dist unif en [-50, 50] clasificados por\n recta con 10% ruido en cada etiqueta, mediante la clasificación que daría f'
title = 'Visualización de realización muestral, de tamaño 100, de v.a. bidimensional de dist. unif. en [-50, 50] en cada componente,\n clasificada utilizando una recta y con 10% ruido en cada etiqueta, vista mediante la clasificación que daría f'

print('Para f(x,y) = (x-10)**2 + (y-20)**2 - 400')
plot_datos_cuad(x, y, f1_plt, title)
print('Accuracy = {}'.format( accuracy(true_y, f1_vct(x[:,0], x[:,1])) ) )

print('Para f(x,y) = 0.5*(x+10)**2 + (y-20)**2 - 400')
plot_datos_cuad(x, y, f2_plt, title)
print('Accuracy = {}'.format( accuracy(true_y, f2_vct(x[:,0], x[:,1])) ) )

print('Para f(x,y) = 0.5*(x-10)**2 - (y+20)**2 - 400')
plot_datos_cuad(x, y, f3_plt, title)
print('Accuracy = {}'.format( accuracy(true_y, f3_vct(x[:,0], x[:,1])) ) )

print('Para f(x,y) = y - 20*x**2 - 5*x + 3')
plot_datos_cuad(x, y, f4_plt, title)
print('Accuracy = {}'.format( accuracy(true_y, f4_vct(x[:,0], x[:,1])) ) )

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON
print('EJERCICIO 2.1: ALGORITMO PERCEPTRON')

'''
   Implementación de algoritmo PLA. 
   
   :param numpy.ndarray datos: matriz muestral de vectores de características con 1s en primera columna
   :param numpy.ndarray label: vector con etiquetas +1 o -1 de la muestra
   :param int max_iter: iteraciones o épocas máximas permitidas
   :param numpy.ndarray vini: valor inicial del vector de pesos
   :return: el último vector de pesos calculado y el número de iteraciones realizadas
'''
def ajusta_PLA(datos, label, max_iter, vini):
    
    w = vini.copy() # guardamos vector de pesos inicial
    wrong_class = True  # suponemos que existe al menos un punto mal clasificado
    it = 0 
    # Mientras no superemos el máximo de iteraciones y haya puntos mal clasificados
    while it < max_iter and wrong_class:
        wrong_class = False # suponemos que no hay puntos mal clasificados
        it += 1
        # Recorremos Dataset; vector caract con 1 al principio y la etiqueta asociada al v. características
        for x, y in zip(datos, label):
            # Si está mal clasificado
            if signo(x.dot(w)) != y:
                w = w + y * x # actualizamos vector pesos desplazando en la dirección correcta
                wrong_class = True # anotamos que había punto mal clasificado
        
    return w, it

print('Apartado a) 1)')

# Guardamos en X la matriz x con la primera columna con 1s
X = np.array([[1, x_t[0], x_t[1]] for x_t in x])

# Caso vector pesos inicial de 0s
iterations = []
w0 = np.zeros(X.shape[1]) # vector inicial pesos 0s
iters_media = 0
# Repetimos 10 veces
for i in range(0, 10):
    w, iters = ajusta_PLA(X, true_y, 1000, w0)
    iterations.append(iters)

print('Promedio de iteraciones necesario para converger con vector de pesos inicial de 0s: {}'.format(np.mean(np.asarray(iterations))))
print('Vector de iteraciones: {}'.format(iterations))
print('Accuracy = {}'.format( accuracy(true_y, f_vct(x[:,0], x[:,1], -w[1]/w[2], -w[0]/w[2])) ))

# Dibujamos la recta obtenida en la última repetición
ptos_eval_recta = np.array([-50.0, 50.0], dtype=np.double) # guardamos valores [-50, 50] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta usada por f para clasificar') # mostramos la recta utilizada para etiquetar
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta obtenida mediante PLA') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[true_y==-1, 0], x[true_y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[true_y==1, 0], x[true_y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Recta de separación usada por f y la obtenida por PLA para el dataset.\n Vector de pesos inicial 0s')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-53, 53)
plt.xlim(-53, 53)
plt.show()

# Caso vector pesos inicial de valores aleatorios en [0,1]
# Caso vector pesos inicial de 0s
iterations = []
iters_media = 0
# Repetimos 10 veces
for i in range(0, 10):
    w0 = np.random.uniform(0, 1, X.shape[1]) # vector pesos inicial de valores aleatorios en [0,1]
    w, iters = ajusta_PLA(X, true_y, 1000, w0)
    iterations.append(iters)
    
print('Promedio de iteraciones necesario para converger con vector de pesos inicial de val. aleatorios en [0,1]: {}'.format(np.mean(np.asarray(iterations))))
print('Vector de iteraciones: {}'.format(iterations))
print('Accuracy = {}'.format( accuracy(true_y, f_vct(x[:,0], x[:,1], -w[1]/w[2], -w[0]/w[2])) ))

# Dibujamos la recta obtenida en la última repetición
ptos_eval_recta = np.array([-50.0, 50.0], dtype=np.double) # guardamos valores [-50, 50] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta usada por f para clasificar') # mostramos la recta utilizada para etiquetar
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta obtenida mediante PLA') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[true_y==-1, 0], x[true_y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[true_y==1, 0], x[true_y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Recta de separación usada por f y la obtenida por PLA para el dataset.\n Vector de pesos inicial valores aleatorios en [0,1]')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-53, 53)
plt.xlim(-53, 53)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Apartado a) 2)
# Ahora con los datos del ejercicio 1.2.b
# Repetimos para el caso de muestra no separable

print('Apartado a) 2)')

# Caso vector pesos inicial de 0s
iterations = []
w0 = np.zeros(X.shape[1]) # vector inicial de pesos 0s
iters_media = 0
# Repetimos 10 veces
for i in range(0, 10):
    w, iters = ajusta_PLA(X, y, 1000, w0)
    iterations.append(iters)

print('Promedio de iteraciones necesario para converger con vector de pesos inicial de 0s: {}'.format(np.mean(np.asarray(iterations))))
print('Vector de iteraciones: {}'.format(iterations))
print('Accuracy = {}'.format( accuracy(true_y, f_vct(x[:,0], x[:,1], -w[1]/w[2], -w[0]/w[2])) ))

# Dibujamos la recta obtenida en la última repetición
ptos_eval_recta = np.array([-50.0, 50.0], dtype=np.double) # guardamos valores [-50, 50] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta usada por f para clasificar') # mostramos la recta utilizada para etiquetar
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta obtenida mediante PLA') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[y==1, 0], x[y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Recta de separación usada por f y la obtenida por PLA para el dataset\n con ruido. Vector de pesos inicial 0s')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-53, 53)
plt.xlim(-53, 53)
plt.show()

# Caso vector pesos inicial de valores aleatorios en [0,1]
# Caso vector pesos inicial de 0s
iterations = []
iters_media = 0
# Repetimos 10 veces
for i in range(0, 10):
    w0 = np.random.uniform(0, 1, X.shape[1]) # vector pesos inicial de valores aleatorios en [0,1]
    w, iters = ajusta_PLA(X, y, 1000, w0)
    iterations.append(iters)
    
print('Promedio de iteraciones necesario para converger con vector de pesos inicial de val. aleatorios en [0,1]: {}'.format(np.mean(np.asarray(iterations))))
print('Vector de iteraciones: {}'.format(iterations))
print('Accuracy = {}'.format( accuracy(true_y, f_vct(x[:,0], x[:,1], -w[1]/w[2], -w[0]/w[2])) ))

# Dibujamos la recta obtenida en la última repetición
ptos_eval_recta = np.array([-50.0, 50.0], dtype=np.double) # guardamos valores [-50, 50] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta usada por f para clasificar') # mostramos la recta utilizada para etiquetar
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta obtenida mediante PLA') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[y==1, 0], x[y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Recta de separación usada por f y la obtenida por PLA para el dataset\n con ruido. Vector de pesos inicial valores aleatorios en [0,1]')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-53, 53)
plt.xlim(-53, 53)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
print('EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT')

# Ein regresión logística
def Ein_RL(X, y, w):
    N = X.shape[0]
    err = 0
    for item, etiq in zip(X,y):
       err += np.log(1 + np.exp(-etiq * w.dot(item)))
       
    err = (1/N) * err
    
    return err

# Gradiente Ein en minibatch (reg logística)
def grad_Ein_minibatch_RL(X, y, w):
    M = X.shape[0]
    gradiente = np.zeros(X.shape[1])

    for item, etiq in zip (X, y):
        gradiente += (etiq * item) / (1 + np.exp(etiq * w.dot(item)))
    
    gradiente = -(1/M) * gradiente
    
    return gradiente

'''
   Implementación de algoritmo de Regresión Logística (SGD). 
   
   :param numpy.ndarray X: matriz muestral de vectores de características con 1s en primera columna
   :param numpy.ndarray y: vector con etiquetas +1 o -1 de la muestra
   :param int max_epocas: épocas máximas permitidas
   :param float learning_rate: tasa de aprendizaje
   :param int minibatches_size: tamaño minibatches
   :param float e_crit_parada: criterio parada para la distancia entre el vector de pesos de una época y el v. pesos de la época anterior
   :return: el último vector de pesos calculado y el número de épocas hasta la finalización del algoritmo
'''
def sgdRL(X, y, max_epocas, learning_rate=0.01, minibatches_size=1, e_crit_parada=0.01):
    w = np.zeros(X.shape[1]) # vector de pesos inicial 0s
    wt = w.copy() # vector de pesos  inicial en la primera época 0s
    wt_1 = [np.inf for i in range(X.shape[1])] # vector de pesos en la época anterior, +inf por no haber época anterior y evitar condición parada
    sample_size = X.shape[0] # tamaño de muestra
    indexes = np.arange(sample_size) # trabajaremos con índices
    # Mezclamos aleatoriamente la muestra (trabajando con índices) por primera vez
    np.random.shuffle(indexes)
    # Marcamos el primer minibatch
    minibatch_init = 0
    minibatch_end = minibatch_init + minibatches_size
    epocas = 0 # contaremos las épocas
    
    # Mientras que no alcancemos el máximo de iteraciones y ||w(t-1)-w(t)|| no alcance la cota
    while epocas < max_epocas and np.linalg.norm(wt_1 - wt) >= e_crit_parada:
        # Se toma el minibatch actual
        minibatch = indexes[minibatch_init:minibatch_end]
        # Se calcula el vector pesos para el minibatch tomado (dando el 'paso de mayor profundidad' en el minibatch)
        w = w - learning_rate * grad_Ein_minibatch_RL(X[minibatch], y[minibatch], w)
        # Se avanza al siguiente minibatch
        minibatch_init += minibatches_size
        minibatch_end += minibatches_size
        # Si ya hemos iterado en todos los minibatches (hemos completado una época)
        if (minibatch_init >= sample_size):
            # actualizamos vectores de pesos de época actual y anterior
            wt_1 = wt.copy()
            wt = w.copy()
            epocas += 1 # incrementamos número de épocas
            # Mezclamos aleatoriamente muestra (trabajando con índices), generando así nuevos minibatches
            np.random.shuffle(indexes)
            # reestablecemos valores para empezar por primer minibatch
            minibatch_init = 0
            minibatch_end = minibatch_init + minibatches_size
            
    return wt, epocas

# Elegimos 100 puntos aleatorios de [0,2]x[0,2]
x = simula_unif(100, 2, [0,2])

# Simulamos recta frontera que utilizaremos para clasificar
a, b = simula_recta([0, 2])
print('Clasificaremos mediante la recta ax + b con a = {}, b = {}'.format(a,b))

# Obtenemos etiquetas para la muestra con la recta
y = f_vct(x[:,0], x[:,1], a, b)

# Generamos gráfica con la muestra etiquetada y la recta utilizada para etiquetar
ptos_eval_recta = np.array([0, 2], dtype=np.double) # guardamos valores [0, 2] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, a*ptos_eval_recta + b, 'k-', label='recta ax+b usada para clasificar') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[y==1, 0], x[y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Realización muestral de tamaño 100 de vector aleatorio bidimensional con\n dist. uniforme [0, 2] en cada componente, clasificada mediante recta ax+b')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-0.1, 2.1)
plt.xlim(-0.1, 2.1)
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
    
X = np.array([[1, x_t[0], x_t[1]] for x_t in x]) # añadimos primera columna de 1s

w, epocas = sgdRL(X, y, 10000)
print('Se ejecuta algoritmo sgdRL y termina depués de {} épocas'.format(epocas))
print('Accuracy = {}'.format( accuracy(y, f_vct(x[:,0], x[:,1], -w[1]/w[2], -w[0]/w[2])) ))
print('Vector de pesos obtenido w: {}'.format(w))
print('Error de la estimación g determinada por w, en la muestra, Ein(g) = {}'.format(Ein_RL(X,y,w)))

# Dibujamos la recta obtenida
ptos_eval_recta = np.array([0, 2], dtype=np.double) # guardamos valores [0, 2] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'k-', label='recta frontera estimada') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x[y==1, 0], x[y==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Realización muestral de tamaño 100 de vector aleatorio bidimensional con dist. U(0,2)\n en cada componente, junto con la recta estimada frontera obtenida mediante sgdRL')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-0.1, 2.1)
plt.xlim(-0.1, 2.1)
plt.show()

# Ahora estimamos Eout
x_test = simula_unif(1000, 2, [0,2])
y_test = f_vct(x_test[:,0], x_test[:,1], a, b)

X_test = np.array([[1, x_t[0], x_t[1]] for x_t in x_test]) # añadimos primera columna de 1s

print('Estimación del error fuera de la muestra: {}'.format(Ein_RL(X_test, y_test, w)))
print('Accuracy = {}'.format( accuracy(y_test, f_vct(x_test[:,0], x_test[:,1], -w[1]/w[2], -w[0]/w[2])) ))

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

# Dibujamos la recta obtenida junto con la original 
ptos_eval_recta = np.array([0, 2], dtype=np.double) # guardamos valores [0, 2] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta frontera estimada') # mostramos la recta utilizada para etiquetar
plt.plot(ptos_eval_recta, a*ptos_eval_recta+b, 'm-', label='recta frontera original') # mostramos la recta utilizada para etiquetar
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x_test[y_test==-1, 0], x_test[y_test==-1, 1], 'r.', label = '-1') # los de etiqueta -1
plt.plot(x_test[y_test==1, 0], x_test[y_test==1, 1], 'b.', label = '1') # los de etiqueta +1
plt.title('Realización muestral de tamaño 1000 de vector aleatorio bidimensional con dist. U(0,2) en\n cada componente, junto con la recta estimada frontera obtenida mediante sgdRL')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Primera coordenada')
plt.ylabel('Segunda coordenada')
plt.ylim(-0.1, 2.1)
plt.xlim(-0.1, 2.1)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Repetimos el experimento 100 veces para obtener promedio de épocas en converger')
print('... Tarda en ejecutar ...')

epocas_p = []
etests = []
eins =[]
for i in np.arange(100):
    # Elegimos 100 puntos aleatorios de [0,2]x[0,2]
    x = simula_unif(100, 2, [0,2])

    # Obtenemos etiquetas
    y = f_vct(x[:,0], x[:,1], a, b)
    X = np.array([[1, x_t[0], x_t[1]] for x_t in x]) # añadimos primera columna de 1s

    w, epocas = sgdRL(X, y, 10000) # Ejecutamos RL guandando w, y epocas
    
    epocas_p.append(epocas)

    # Generamos muestra de test y obtenemos sus etiquetas mediante f
    x_test = simula_unif(1000, 2, [0,2])
    y_test = f_vct(x_test[:,0], x_test[:,1], a, b)

    X_test = np.array([[1, x_t[0], x_t[1]] for x_t in x_test]) # añadimos primera columna de 1s

    etests.append(Ein_RL(X_test, y_test, w)) # Guardamos las estimaciones de eout
    eins.append(Ein_RL(X, y, w))

print('Número de épocas promedio en converger: {}'.format( np.mean(np.asarray(epocas_p)) ))
print('Estimación promedio del error fuera de la muestra: {}'.format( np.mean(np.asarray(etests)) ))
print('Estimación promedio del error dentro de la muestra: {}'.format( np.mean(np.asarray(eins)) ))
print('Épocas obtenidas ordenadas de menor a mayor:')
print(np.sort(np.asarray(epocas_p)))

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos
print('BONUS: Clasificación de Dígitos')

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

print('Tamaño muestra de entrenamiento: {}'.format(x.shape[0]))
print('Tamaño muestra de test: {}'.format(x_test.shape[0]))

plt.plot(x[y==-1, 1], x[y==-1, 2], 'r.', label = '4') 
plt.plot(x[y==1, 1], x[y==1, 2], 'b.', label = '8')
plt.title('Digitos Manuscritos (TRAINING)')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.xlim(0,1)
plt.show()

plt.plot(x_test[y_test==-1, 1], x_test[y_test==-1, 2], 'r.', label = '4') 
plt.plot(x_test[y_test==1, 1], x_test[y_test==1, 2], 'b.', label = '8')
plt.title('Digitos Manuscritos (TEST)')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.xlim(0,1)
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 
print('Pseudoinversa')
# Pseudoinversa	
'''
   Implementación de método por pseudoinversa.
   
   :param numpy.ndarray X: matriz muestral de vectores de características 1a columna de 1s
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

w_lin = pseudoinverse(x, y)

print('Vector de pesos obtenido: {}'.format(w_lin))

# Dibujamos la recta obtenida en muestra entrenamiento
ptos_eval_recta = np.array([0.1, 0.45], dtype=np.double) # guardamos valores [0.1, 0.45] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, -(w_lin[0]+w_lin[1] * ptos_eval_recta) / w_lin[2], 'g-', label='recta frontera Pseudoinversa') # mostramos la recta frontera obtenida por Pseudoinversa
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 1], x[y==-1, 2], 'r.', label = '4') # los de etiqueta -1 son 4
plt.plot(x[y==1, 1], x[y==1, 2], 'b.', label = '8') # los de etiqueta +1 son 8
plt.title('Dígitos Manuscritos (TRAINING) junto con recta frontera obtenida por Pseudoinversa')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.xlim(0,1)
plt.show()

acc = accuracy(y, f_vct(x[:,1], x[:,2], -w_lin[1]/w_lin[2], -w_lin[0]/w_lin[2])) 
print('Accuracy = {}'.format( acc ))
print('Error de clasificación en la muestra entrenamiento: {}'.format(1 - acc))

# Dibujamos la recta obtenida en muestra entrenamiento
ptos_eval_recta = np.array([0.1, 0.45], dtype=np.double) # guardamos valores [0.1, 0.45] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, -(w_lin[0]+w_lin[1] * ptos_eval_recta) / w_lin[2], 'g-', label='recta frontera Pseudoinversa') # mostramos la recta frontera obtenida por Pseudoinversa
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x_test[y_test==-1, 1], x_test[y_test==-1, 2], 'r.', label = '4') # los de etiqueta -1 son 4
plt.plot(x_test[y_test==1, 1], x_test[y_test==1, 2], 'b.', label = '8') # los de etiqueta +1 son 8
plt.title('Dígitos Manuscritos (TEST) junto con recta frontera obtenida por Pseudoinversa')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.xlim(0,1)
plt.show()

acc = accuracy(y_test, f_vct(x_test[:,1], x_test[:,2], -w_lin[1]/w_lin[2], -w_lin[0]/w_lin[2])) 
print('Accuracy = {}'.format( acc ))
print('Error de clasificación en la muestra test: {}'.format(1 - acc))

input("\n--- Pulsar tecla para continuar ---\n")
print('PLA-Pocket')

'''
   Error de clasificación, devuelve la proporción de puntos mal clasificados. 
   
   :param numpy.ndarray X: matriz muestral de vectores de características con 1s en primera columna
   :param numpy.ndarray y: vector con etiquetas +1 o -1 de la muestra
   :param numpy.ndarray w: vector de pesos 
   :return: proporción de puntos mal clasificados
'''
def Ein_PLA_Pocket(X, y, w):
    N = X.shape[0] # tamaño muestra
    wrong_class = 0 # contaremos los puntos mal clasificados
    # Recorremos dataset
    for x, et in zip(X, y):
        # contamos los puntos mal clasificados
        if signo(x.dot(w)) != et:
            wrong_class += 1
        
    return wrong_class / N

#POCKET ALGORITHM
'''
   Implementación de algoritmo PLA-Pocket. 
   
   :param numpy.ndarray X: matriz muestral de vectores de características con 1s en primera columna
   :param numpy.ndarray y: vector con etiquetas +1 o -1 de la muestra
   :param numpy.ndarray w0: vector de pesos inicial
   :param int max_epocas: épocas máximas permitidas
   :return: vector de pesos óptimo calculado
'''
def PLA_Pocket(X, y, w0, max_epocas):
    w = w0.copy()
    w_opt = w.copy() # vector de pesos óptimo
    ep = 0 # controlamos el número de épocas
    # controlaremos el error óptimo
    err_opt = Ein_PLA_Pocket(X, y, w0) # asignamos error óptimo al obtenido con w0
    # Mientras no superemos el máximo de épocas permitidas
    while ep < max_epocas:
        # Recorremos dataset
        for x, et in zip(X, y):
            # Aplicamos criterio de adaptación de PLA
            if signo(x.dot(w)) != et:
                w = w + et * x
        # Evaluamos Ein(w) (error de clasificación en la muestra para v. pesos w) 
        ein = Ein_PLA_Pocket(X, y, w) # calculamos error que da w
        # Si el error en la muestra con w es menor que el óptimo hasta ahora
        if ein < err_opt:
            err_opt = ein # actualizamos error óptimo
            w_opt = w.copy() # guardamos el nuevo vector de pesos óptimo
        ep += 1 # incrementamos las épocas
        
    return w_opt
        
w = PLA_Pocket(x, y, w_lin, 1000)
print('Vector de pesos obtenido: {}'.format(w))

# Dibujamos la recta obtenida en muestra entrenamiento
ptos_eval_recta = np.array([0.1, 0.45], dtype=np.double) # guardamos valores [0.1, 0.45] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta frontera PLA-Pocket') # mostramos la recta frontera obtenida por PLA-Pocket
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x[y==-1, 1], x[y==-1, 2], 'r.', label = '4') # los de etiqueta -1 son 4
plt.plot(x[y==1, 1], x[y==1, 2], 'b.', label = '8') # los de etiqueta +1 son 8
plt.title('Dígitos Manuscritos (TRAINING) junto con recta frontera\n obtenida por PLA-Pocket')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.xlim(0,1)
plt.show()

acc_training = accuracy(y, f_vct(x[:,1], x[:,2], -w[1]/w[2], -w[0]/w[2])) 
print('Error de clasificación en la muestra entrenamiento: {}'.format(1 - acc_training))
print('Accuracy = {}'.format( acc_training ))

# Dibujamos la recta obtenida en muestra entrenamiento
ptos_eval_recta = np.array([0.1, 0.45], dtype=np.double) # guardamos valores [0.1, 0.45] que utilizaremos para la gráfica de la recta
plt.plot(ptos_eval_recta, -(w[0]+w[1] * ptos_eval_recta) / w[2], 'g-', label='recta frontera PLA-Pocket') # mostramos la recta frontera obtenida por PLA-Pocket
# Mostramos los elementos de la muestra ya etiquetados
plt.plot(x_test[y_test==-1, 1], x_test[y_test==-1, 2], 'r.', label = '4') # los de etiqueta -1 son 4
plt.plot(x_test[y_test==1, 1], x_test[y_test==1, 2], 'b.', label = '8') # los de etiqueta +1 son 8
plt.title('Dígitos Manuscritos (TEST) junto con recta frontera\n obtenida por PLA-Pocket')
plt.legend(loc='lower right', prop={'size': 7.2})
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.xlim(0,1)
plt.show()

acc_test = accuracy(y_test, f_vct(x_test[:,1], x_test[:,2], -w[1]/w[2], -w[0]/w[2])) 
print('Error de clasificación en la muestra test: {}'.format(1 - acc_test))
print('Accuracy = {}'.format( acc_test ))


input("\n--- Pulsar tecla para continuar ---\n")


# COTA SOBRE EL ERROR
# Devuelve la cota de error fuera de la muestra, Eout(g), basada en Ein(g)
def cota_Eout_Ein(ein_g, N, H_size, delta):
    return ein_g + np.sqrt( (1/(2*N)) * np.log((2*H_size) / delta)  )

print('Cota Eout(g) basada en Ein(g): {}'.format(cota_Eout_Ein(1-acc_training, x.shape[0], 2**(3*64), 0.05)))      

print('Cota Eout(g) basada en Etest(g): {}'.format(cota_Eout_Ein(1-acc_test, x_test.shape[0], 1, 0.05)))            