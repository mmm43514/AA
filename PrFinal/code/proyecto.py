# -*- coding: utf-8 -*-
"""
PROYECTO
Nombre Estudiante 1: Mario Muñoz Mesa
Nombre Estudiante 2: Pedro Ramos Suárez
"""
# Primero se presenta ejecución con VarianceThreshold(0.05) en preprocesado y luego
# con VarianceThreshold(0).
import numpy as np
import seaborn as sns  # librería para visualización de datos

from matplotlib import cm
from timeit import default_timer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve

sns.set_style(style='white')
# sns.set_theme() # tema por defecto
palette = sns.color_palette("bright", 11)  # paleta colores bright, un color por clase

SEED = 1
WAIT = True
VISUALIZE = True
VISUALIZE_DIGITS = True
DIGITS = 3

# Fijamos semilla
np.random.seed(SEED)

# Función para parar
def wait():
    if WAIT:
        input("\n--- Pulsar enter para continuar ---\n")

# Función para leer los datos
def readData(filename):
    data = np.genfromtxt(filename, dtype=np.double, delimiter=',')
    return data[:, :-1], data[:, -1].astype(int)

# Muestra un scatter plot de puntos que pueden estar etiquetados por clases
def scatterPlot(X, y, axis, title=None, cmap=cm.tab20):
    # Información general de la gráfica
    plt.figure(figsize=(8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if title is not None:
        plt.title(title)

    # Calculamos los límites de cada eje
    xMin, xMax = np.min(X[:, 0]), np.max(X[:, 0])
    yMin, yMax = np.min(X[:, 1]), np.max(X[:, 1])
    scaleX = (xMax - xMin) * 0.1
    scaleY = (yMax - yMin) * 0.1
    plt.xlim(xMin - scaleX, xMax + scaleX)
    plt.ylim(yMin - scaleY, yMax + scaleY)

    # Mostramos el scatter plot con la clase a la que pertenecen
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.6)
    if y is not None:
        plt.legend(*scatter.legend_elements(), title="Clases", loc="upper right")

    plt.show()

# Aplica TSNE para representar el conjunto en 2 dimensiones
def plotTSNE(X, y):
    X_embedded = TSNE().fit_transform(X)
    scatterPlot(X_embedded, y, axis=["1a componente", "2da componente"], title="Visualización con TSNE")

# Función para dibujar instancia
def plot_optdig_inst(x,label):
    plt.matshow(np.reshape(x, (8,8)), cmap='binary')
    plt.colorbar()
    plt.title("Dígito {}".format(label))

    plt.show()
    
# Función para dibujar número de instancias de vect caract para cada clase
def plot_class_insts(y, tit):
    inst_class = np.bincount(y)
    print('Vector con número de instancias por clase en {}: '.format(tit))
    print(inst_class)
    plt.bar(np.unique(y_train), inst_class)
    plt.xlabel('Dígito')
    plt.ylabel('Número de elementos en la muestra')
    plt.title('Conteo por clase en {}'.format(tit))
    plt.xticks([i for i in range(10)])
    plt.show()
   
# Comprobamos si hay coef de correlación mayor a thrshld
def check_high_corr(corr_matrix, thrshld):
    feat_to_remove =[]
    # Recorremos matriz de coef correlación
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[0]):
            # tomamos solo una parte iqda de la matriz (matriz simétrica) y si tenemos coef corr > thrshld
            if i > j and corr_matrix[i][j] > thrshld: # Si tenemos coef > thrshld
                # guardamos feature en lista
                feat_to_remove.append(i)
                print(corr_matrix[i][j])

    feat_to_remove = np.unique(np.array(feat_to_remove)) # quitamos repetidos
    if len(feat_to_remove)==0:
        print('No hay coef correlación lineal con valor mayor a {}'.format(thrshld))
    else:
        print('Eliminaremos los características que tengan |coef correlación| > {} con otras características, en concreto: '.format(thrshld))
        print(feat_to_remove)
        return(feat_to_remove)
    
# Función para estimar el número de neuronas en capas ocultas
def estimate_hidden_layer_sizes(h_l, pipeline):
    models = [{"model": [MLPClassifier(#hidden_layer_sizes a estimar
                                   activation='relu', # elegimos función activación relu por ventajas ante tanh y sigmoide
                                   solver='adam', # elegimos método adam, robusto y eficiente, no requiere afinar tanto parámetros como sgd
                                   alpha=0.0001, # parámetro de regularización l2
                                   batch_size=128, # elegimos tamaño de batch arbitrario, pensamos que es un tamaño razonable para el tamaño de nuestro dataset aunque no podemos saber cómo se comportará si no es experimentalmente
                                   # es candidato a hiperparámetro a estimar por cv, nosotros fijamos uno por no hacer cv con mucha carga
                                   learning_rate='adaptive', # solo usado cuando solver='sgd', no es nuestro caso
                                   learning_rate_init=0.001, # elegimos learning rate inicial recomendado (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
                                   power_t=0.5, # solo usado cuando solver='sgd', no es nuestro caso
                                   max_iter=250,  # número máximo de épocas arbitrario pero que muestra convergencia
                                   shuffle=True,  # mezclamos después de cada época
                                   random_state=SEED, # para tener reproducibilidad de los resultados
                                   tol=0.001, # tolerancia de parada, si no mejoramos una milésima tras n_iter_no_change épocas entonces paramos, pensamos que es un valor razonable
                                   verbose=False, # no queremos mensajes
                                   warm_start=False, # no reutilizamos ninguna solución anterior durante la validación cruzada
                                   momentum=0.9, # solo usado cuando solver='sgd', no es nuestro caso
                                   nesterovs_momentum=True, # solo usado cuando solver='sgd', no es nuestro caso
                                   early_stopping=False, # pues consideramos que tenemos un bajo tamaño de entrenamiento
                                   validation_fraction=0.1, # no nos importa valor pues no usamos early_stopping
                                   beta_1=0.9, # elegimos tasa exponential decay para primer momento recomendada (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
                                   beta_2=0.999, # elegimos tasa exponential decay para segundo momento recomendada (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
                                   epsilon=1e-08, # valor numerical stability para adam recomendado (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
                                   n_iter_no_change=10, # damos hasta 10 épocas para mejorar error en 0.001, si no se mejora pararemos
                                   max_fun=15000)], # solo usado cuando solver='lbfgs', no es nuestro caso
             "model__hidden_layer_sizes": h_l}
        ]

    print("Ajustaremos hiperparámetro hidden_layer_sizes mediante 10-fold cross validation")
    
    winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                            models, # modelos
                                            scoring = 'accuracy', # métrica a valorar
                                            n_jobs = -1, # máxima paralelización posible en ejecución
                                            refit = False, # no entrenamos el modelo en toda la muestra, solo estamos buscando hiperparámetro hidden_layer_sizes
                                            cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                            verbose = 0, # no nos interesan mensajes
                                            # pre_dispatch = njobs, lanzamos n_jobs procesos
                                            # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                            # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                            )
    
    print("Ejecutando GridSearchCV (tarda en ejecutar): búsqueda hiperparámetro hidden_layer_sizes\n")
    winner_model.fit(X_train, y_train)
    
    print("Vector con la media de accuracy por val. cruzada de cada modelo")
    print(winner_model.cv_results_['mean_test_score'])
    
    print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
    print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
    
    plt.plot(np.unique(h_l), winner_model.cv_results_['mean_test_score'], color='k', ls='-')
    plt.title('Accuracy para parámetros hidden_layer_size')
    plt.xlabel('hidden_layer_size')
    plt.ylabel('cross validation accuracy')
    plt.show()
    
# Función para estimar el número de árboles de decisión para Random Forest
def estimate_n_estimators(n_estimators, pipeline):
    print("Ajustaremos el hiperparámetros n estimators mediante 10-fold cross validation")

    models = [{"model": [RandomForestClassifier(
        criterion='gini',   # usamos gini porque es mas rapido y nos da resultados similares
        max_depth=None,     # para que explore la profundidad maxima del arbol
        min_samples_split=2,  # número mínimo de muestras para dividir un nodo, con 2 divide siempre que no sean hojas
        min_samples_leaf=1,     # una hoja se considera cuando solo es una muestra
        min_weight_fraction_leaf=0.0,   # para que todas las hojas tenga el mismo peso
        max_features='auto',    # utiliza la raiz cuadrada del número de características
        max_leaf_nodes=None,    # para que utilice todos los arboles
        min_impurity_decrease=0.0,  # para que expanda todos los nodos
        min_impurity_split=None,    # para que expanda todos los nodos
        bootstrap=True,     # como hemos visto en teoría, mejoramos los resultados usando bootstrap
        oob_score=False,     # solo queremos que use los datos de la muestra
        n_jobs=-1,      # para que utilice todos los cores del procesador
        random_state=SEED,  # para tener reproducibilidad de los resultados
        verbose=0,  # no queremos mensajes
        warm_start=False,   # para que cree el arbol de cero
        class_weight=None,  # para que todas las clases tengan el mismo peso
        ccp_alpha=0.0,      # en un principio no queremos poda
        max_samples=None)],     # para que utilize todos los datos
        "model__n_estimators": n_estimators}]

    winner_model = model_selection.GridSearchCV(pipeline,  # pipeline
                                                models,  # modelos
                                                scoring='accuracy',  # métrica a valorar
                                                n_jobs=-1,  # máxima paralelización posible en ejecución
                                                refit=False,
                                                # no entrenamos el modelo en toda la muestra, solo estamos buscando los hiperparámetros max depth y n estimators
                                                cv=10,
                                                # 10-fold cross validation (estratificado, conserva proporción de clases)
                                                verbose=0,
                                                # no mostramos toda la información posible para mayor claridad en los resultados impresos por pantalla
                                                # pre_dispatch=njobs, lanzamos n_jobs procesos
                                                # error_score=np.nan , asignamos np.nan a error_score si se produce error
                                                # return_train_score=False , es costoso computacionalmente y no lo necesitamos
                                                )

    print("Ejecutando GridSearchCV (tarda en ejecutar): búsqueda hiperparámetro n estimators\n")
    winner_model.fit(X_train, y_train)

    print("Vector con la media de accuracy por val. cruzada de cada modelo")
    print(winner_model.cv_results_['mean_test_score'])

    print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
    print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(
        winner_model.best_score_))

    plt.plot(np.unique(n_estimators), winner_model.cv_results_['mean_test_score'], color='k', ls='-')

    plt.title('Accuracy para parámetro n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('cross validation accuracy')
    plt.show()
    
# Función para estimar gamma en modelo SVM
def estimate_gamma_svm(gam, pipeline, c):
    models = [{"model": [SVC(C=c, # parámetro c
                         kernel='rbf', # kernel Radial Basis Function
                         degree=3, # solo para kernel poly, no es nuestro caso
                         #gamma, lo pasamos como argumento
                         coef0=0.0, # solo para kernel poly, no es nuestro caso
                         shrinking=True, # suponemos número alto de iteraciones, nos podrá ayudar a reducir tiempo de cómputo
                         probability=False, # consideramos que no es necesario y nos aumentaría coste computacional
                         tol=0.001, # tolerancia para criterio de parada, hasta milésima
                         cache_size=200, # puede mejorar tiempo ejecución para problemas de muchos datos, en nuestro caso consideramos que son pocos datos y que con 200MB sería suficiente
                         class_weight=None,  # suponemos todas las clases con peso 1 pues tenemos clases balanceadas
                         verbose=False, # no queremos mensajes
                         max_iter=- 1, # no establecemos criterio de parada por iteraciones
                         decision_function_shape='ovr',  # devolvemos función decisión one vs rest 
                         break_ties=False, # no consideramos casos de empates, y ahorraremos considerablemente en coste computacional
                         random_state=SEED)],     # para que utilize todos los datos
        "model__gamma": gam # param que pasamos como argumento
        }]

    winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = False, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )

    #print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
    winner_model.fit(X_train, y_train)
    #print("Vector con la media de accuracy por val. cruzada de cada modelo en orden de ejecución")
    #print(winner_model.cv_results_['mean_test_score'])
    #print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
    #print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
    return winner_model.cv_results_['mean_test_score']

# Función para estimar C en modelo SVM
def estimate_c_svm(c, a, pipeline):
    models = [{"model": [SVC(#C=c, 
                         kernel='rbf',
                         degree=3, 
                         gamma=a, 
                         coef0=0.0, 
                         shrinking=True, 
                         probability=False, 
                         tol=0.001, 
                         cache_size=200, 
                         class_weight=None, 
                         verbose=False, 
                         max_iter=- 1, 
                         decision_function_shape='ovr', 
                         break_ties=False, 
                         random_state=SEED)],     # para que utilize todos los datos
               "model__C": c}]
    winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = False, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )
    print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
    winner_model.fit(X_train, y_train)
    print("Vector con la media de accuracy por val. cruzada de cada modelo en orden de ejecución")
    print(winner_model.cv_results_['mean_test_score'])
    print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
    print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
    return winner_model.cv_results_['mean_test_score']
    
# Código plot_learning_curve copiado de https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# CLASIFICACIÓN - Optical recognition of handwritten digits
print('CLASIFICACIÓN - Optical recognition of handwritten digits')

# Leemos los datos
X_train, y_train = readData("datos/optdigits.tra") # Datos de entrenamiento
X_test, y_test = readData("datos/optdigits.tes") # Datos de test
'''
X=np.concatenate([X_train, X_test], axis=0)
y=np.concatenate([y_train, y_test], axis=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.32, shuffle=True, stratify=y, random_state = 1)
'''
# Extraemos número instancias y tamaño vectores 
N, d = X_train.shape
print('Dataset con {} instancias de vectores de características {}-dimensionales'.format(N,d))

print('Número de instancias en training: {}'.format(X_train.shape[0]))
print("Proporción de training: {:.3f}%".format((X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100)))
print('Número de instancias en test: {}'.format(X_test.shape[0]))
print("Proporción de test: {:.3f}%".format((X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100)))

# Vemos el número de instancias por clase en training y test para valorar si tenemos clases balanceadas
plot_class_insts(y_train, "training")
plot_class_insts(y_test, "test")

wait()

# Representamos algunos dígitos con sus etiquetas
if VISUALIZE_DIGITS:
    random = np.random.choice(range(len(X_train)), DIGITS, replace=False)
    for i in random:
        plot_optdig_inst(X_train[i], y_train[i])

    wait()

# Representamos los datos con TSNE
if VISUALIZE:
    print("Visualización de los datos de entrenamiento con TSNE")
    plotTSNE(X_train, y_train)

    wait()

# Quitamos caract con varianza 0 para poder computar la matriz de coef de correlación lineal
X_varth = Pipeline([('varth', VarianceThreshold())]).fit_transform(X_train, y_train) 
corr_matrix = np.abs( np.corrcoef(np.transpose(X_varth)) )
# Visualizamos matriz de coef. correlación mediante heatmap
ax = sns.heatmap(corr_matrix, cmap="Reds")
plt.title('Heatmap de la matriz de coeficientes de correlación\n entre las variables (en valor absoluto)')
plt.show()

# Comprobamos si tenemos caract con alto coef correlación lineal, 0.95
check_high_corr(corr_matrix, 0.95)

# Calculamos las varianzas de las características
vars = []
for i in range(X_train.shape[1]):
    vars.append(np.var(X_train[:,i]))

# Dibujamos las varianzas de cada característica
plt.plot([i+1 for i in range(X_train.shape[1])], vars, color='k', ls='-')
plt.title('Varianzas por característica')
plt.xlabel('Característica')
plt.ylabel('Varianza')
plt.show()

# Eliminaremos features con varianza < 0.05
feat_low_var = []
print("Features a eliminar")
for j in range(len(vars)):
    if vars[j]<0.05:
        print("Feature {} con varianza={} < 0.05 y media {}".format(j, vars[j], np.mean(X_train[:,j] )))
        feat_low_var.append(j)

print("Número de características original: {}, después de eliminar features: {}".format(X_train.shape[1], X_train.shape[1]-len(feat_low_var)))

# Mostramos percentiles 99 caract con varianza < 0.05
for f in feat_low_var:
    print("Percentil 99 de feature {} es {}".format(f, np.percentile(X_train[:,f],99)))
# Mostramos percentil 98 de característica 47
print("Percentil 98 de feature {} es {}".format(47, np.percentile(X_train[:,f],99)))

wait()

#####################################################################################
################################### MODELO LINEAL ###################################
#####################################################################################

print("MODELO LINEAL: Regresión Logística multiclase one-vs-rest")
pipeline = Pipeline([
    ('VarThr', VarianceThreshold(threshold = 0.05)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('PolFeat', preprocessing.PolynomialFeatures(degree=2, # Transformamos vectores caract. a vect. caract. con combinaciones de producto caract con grado menor o igual que 2
                                                 interaction_only=False, # pues queremos todas las combinaciones
                                                 include_bias = False)), # no añadimos columna de 1s pues el siguiente paso es normalizar,  el sesgo lo añadiremos en las funciones de nuestros algoritmos de aprendizaje. 
    ('Post PolFeat StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                                 with_mean=True, # centramos los datos antes de escalar
                                                                 with_std = True)), # queremos tener varianza 1
    ("model", SGDClassifier())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

# los modelos de búsqueda serán SGDClassifier con cada valor para regularización y Perceptron con cada valor para regularización, en total 6 modelos
models = [
        {"model": [SGDClassifier(
            loss = 'log', # función de pérdida de regresión logística
            penalty = 'l2', # utilizaremos regularización l2 (reg. Ridge)
            # alpha  constante del término de regularización (probaremos distintos valores mediante 10-fold cross validation)
            fit_intercept = True, # añadimos sesgo o intercept pues nuestra matriz aún no tiene columna de 1s 
            max_iter = 180, # Número máximo de iteraciones arbitrario
            #tol  Tolerancia para criterio de parada por tolerancia (parar si loss > best_loss - tol tras n_iter_no_change épocas seguidas)
            # el criterio valora si el error no mejora en tol el mejor error hasta el momento. En nuestro caso esta condición por tolerancia se usará para learning_rate adaptativo
            shuffle = True, # mezclamos después de cada época,
            n_jobs = -1, # máxima paralelización posible en ejecución
            random_state = SEED, # para tener reproducibilidad de los resultados
            learning_rate = 'adaptive', # si no se mejora resultado por criterio tolerancia (loss > best_loss - tol) tras n_iter_no_change épocas seguidas, entonces cambiamos learning rate por (learning rate)/5
                                        # queremos evitar oscilación
            eta0 = 0.05, # learning rate inicial arbitrario
            early_stopping = False, # False pues no queremos reservar más datos para validación, consideramos nuestro dataset pequeño y que no conviene quitar más datos de training
            n_iter_no_change = 5, # cada 5 épocas sin mejora en crit. tolerancia se realizará la adaptación del learning rate (learning_rate='adaptive')
            class_weight = None, # se interpreta que todas las clases tienen peso 1, se elige así pues nuestro problema tiene clases balanceadas
            average = False, # no nos interesa obtener media de pesos
            verbose = 0, # no nos interesan mensajes 
            warm_start = False, # no reutilizamos ninguna solución anterior durante la validación cruzada
            l1_ratio = 0 # 0 corresponde a l2 penalty, no se usará pues solo se usa si learning_rate = 'elasticnet'
            )], # máxima paralelización posible
         "model__alpha": [0.000001, 0.0001, 0.001], # probamos valores de regularización (hiperparámetro)
         "model__tol": [0.0001, 0.001, 0.01]} # probamos distintos valores de tolerancia (hiperparámetro)
        ]

print("Buscaremos la mejor hipótesis mediante 10-fold cross validation")

winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)

print("Vector con la media de accuracy por val. cruzada de cada modelo en orden de ejecución")
print(winner_model.cv_results_['mean_test_score'])
print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format( winner_model.score(X_test, y_test)))


wait()

######################################################################################
################################ PERCEPTRON MULTICAPA ################################
######################################################################################
print("Perceptron Multicapa")

# Vamos a buscar parámetro hidden_layer_sizes por validación cruzada

pipeline = Pipeline([
    ('VarThr', VarianceThreshold(threshold = 0.05)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                    with_mean=True, # centramos los datos antes de escalar
                                                    with_std = True)), # queremos tener varianza 1
    ("model", MLPClassifier())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

h_l = [(i,i) for i in range(50,105,5)]
estimate_hidden_layer_sizes(h_l, pipeline)

h_l=[(i,i) for i in range(90,101,1)]
estimate_hidden_layer_sizes(h_l, pipeline)

#h_l = [(94,94)] #hiperparámetro ganador
models = [{"model": [MLPClassifier(
        hidden_layer_sizes=(94,94), # hiperparámetro estimado por validación cruzada
        activation='relu', # elegimos función activación relu por ventajas ante tanh y sigmoide
        solver='adam', # elegimos método adam, robusto y eficiente, no requiere afinar tanto parámetros como sgd
        #alpha parámetro reg. l2 a estimar
        batch_size=128, # elegimos tamaño de batch arbitrario, pensamos que es un tamaño razonable para el tamaño de nuestro dataset aunque no podemos saber cómo se comportará si no es experimentalmente
        # es candidato a hiperparámetro a estimar por cv, nosotros fijamos uno por no hacer cv con mucha carga
        learning_rate='adaptive', # solo usado cuando solver='sgd', no es nuestro caso
        learning_rate_init=0.001, # elegimos learning rate inicial recomendado (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        power_t=0.5, # solo usado cuando solver='sgd', no es nuestro caso
        max_iter=250,  # número máximo de épocas arbitrario pero que muestra convergencia
        shuffle=True,  # mezclamos después de cada época
        random_state=SEED, # para tener reproducibilidad de los resultados
        tol=0.001, # tolerancia de parada, si no mejoramos una milésima tras n_iter_no_change épocas entonces paramos, pensamos que es un valor razonable
        verbose=False, # no queremos mensajes
        warm_start=False, # no reutilizamos ninguna solución anterior durante la validación cruzada
        momentum=0.9, # solo usado cuando solver='sgd', no es nuestro caso
        nesterovs_momentum=True, # solo usado cuando solver='sgd', no es nuestro caso
        early_stopping=False, # pues consideramos que tenemos un bajo tamaño de entrenamiento
        validation_fraction=0.1, # no nos importa valor pues no usamos early_stopping
        beta_1=0.9, # elegimos tasa exponential decay para primer momento recomendada (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        beta_2=0.999, # elegimos tasa exponential decay para segundo momento recomendada (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        epsilon=1e-08, # valor numerical stability para adam recomendado (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        n_iter_no_change=10, # damos hasta 10 épocas para mejorar error en 0.001, si no se mejora pararemos
        max_fun=15000)], # solo usado cuando solver='lbfgs', no es nuestro caso
        "model__alpha": [0.00001,0.0001,0.001]} # parámetro de reg. l2 para cross validation
    ]

print("Buscamos mejor modelo mediante 10-fold cross validation")
winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )


print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)

print("Vector con la media de accuracy por val. cruzada de cada modelo")
print(winner_model.cv_results_['mean_test_score'])
print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format( winner_model.score(X_test, y_test)))


wait()

######################################################################################
################################### RANDOM FOREST ####################################
######################################################################################

print("Random Forest")

# Vamos a buscar parámetro n_estimators por validación cruzada

p = Pipeline([
    ('VarThr', VarianceThreshold(threshold=0.05)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('Post PolFeat StandardScaler', preprocessing.StandardScaler(copy=True,  # no nos importa que se haga copia
                                                                 with_mean=True,  # centramos los datos antes de escalar
                                                                 with_std=True)),  # queremos tener varianza 1
    ("model",
     RandomForestClassifier())])  # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn


n_est = [i for i in range(50, 501, 50)]

estimate_n_estimators(n_est, p)     # 450

n_est = [i for i in range(400, 501, 25)]

estimate_n_estimators(n_est, p)    # 450

n_est = [i for i in range(425, 476, 5)]

estimate_n_estimators(n_est, p)     # 455

n_est = [i for i in range(450, 459, 1)]

estimate_n_estimators(n_est, p)     # 453

#n_est = 453

models = [{"model": [RandomForestClassifier(
        n_estimators=453,  # usamos el número de estimadores calculados
        criterion='gini',   # usamos gini porque es mas rapido y nos da resultados similares
        max_depth=None,     # para que explore la profundidad maxima del arbol
        min_samples_split=2,  # número mínimo de muestras para dividir un nodo, con 2 divide siempre que no sean hojas
        min_samples_leaf=1,     # una hoja se considera cuando solo es una muestra
        min_weight_fraction_leaf=0.0,   # para que todas las hojas tenga el mismo peso
        max_features='auto',    # utiliza la raiz cuadrada del número de características
        max_leaf_nodes=None,    # para que utilice todos los arboles
        min_impurity_decrease=0.0,  # para que expanda todos los nodos
        min_impurity_split=None,    # para que expanda todos los nodos
        bootstrap=True,     # como hemos visto en teoría, mejoramos los resultados usando bootstrap
        oob_score=False,     # solo queremos que use los datos de la muestra
        n_jobs=-1,      # para que utilice todos los cores del procesador
        random_state=SEED,  # para tener reproducibilidad de los resultados
        verbose=0,  # no queremos mensajes
        warm_start=False,   # para que cree el arbol de cero
        class_weight=None,  # para que todas las clases tengan el mismo peso
        #ccp_alpha grado de penalización por complejidad. Cuanto más alto más podado. Parámetro para regularizar y evitar sobreajuste. Será hiperparámetro a estimar
        max_samples=None)],     # para que utilize todos los datos
        "model__ccp_alpha": [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]} # hiperparámetro a estimar
]

print("Buscamos mejor modelo mediante 10-fold cross validation")
winner_model = model_selection.GridSearchCV(p,  # pipeline
                                            models,  # modelos
                                            scoring='accuracy',  # métrica a valorar
                                            n_jobs=-1,  # máxima paralelización posible en ejecución
                                            refit=True,
                                            # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                            cv=10,
                                            # 10-fold cross validation (estratificado, conserva proporción de clases)
                                            verbose=0,  # no mostramos toda la información posible para mayor claridad en los resultados impresos por pantalla
                                            # pre_dispatch = njobs, lanzamos n_jobs procesos
                                            # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                            # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                            )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)
print("Vector con la media de accuracy por val. cruzada de cada modelo")
print(winner_model.cv_results_['mean_test_score'])
print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format(winner_model.score(X_test, y_test)))

wait()

############################################################################
################################### SVM ####################################
############################################################################

print("Support Vector Machine")

p = Pipeline([
    ('VarThr', VarianceThreshold(threshold = 0.05)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                    with_mean=True, # centramos los datos antes de escalar
                                                    with_std = True)), # queremos tener varianza 1
    ("model", SVC())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

print("Estimación hiperparámetro gamma para cada c entre 0.001 y 1000 (tomamos 7 valores en escala logarítmica), hacemos búsqueda dicotómica")
# Estimación hiperparámetro gamma para cada c hacemos búsqueda dicotómica
best_results_m=[] # matriz con filas de mejor resultado (gamma y mejor accuracy) por cada param c
track_g_m =[] # matriz con seguimiento del valor de accuracy al mover gamma, una fila para cada c
c_params=np.logspace(-3,3,7) # parámetros de regularización
# para cada parámetro C
for c in c_params:
    a=0.001 # valor inferior para gamma
    b=1000 # valor superior para gamma
    res=estimate_gamma_svm([a,(a+b)/2,b], p, c) # obtengo accuracies por cv para cada valor
    track_g=[]
    track_g.append([(a+b)/2, res[1]]) # iremos guardando  progresión aumento accuracy
    b_r=[[a,(a+b)/2,b][np.argmax(res)], np.max(res)] # iremos controlando el mejor resultado, guardamos valor gamma y mejor accuracy
    # buscaremos hasta dos cifras decimales fijas y estemos aproximando milésimas
    while np.abs(res[0]-res[1])>=0.001 or np.abs(res[1]-res[2])>=0.001:
        # si hay mejor resultado por la izquierda
        if res[0]>res[2]:
            # nos vamos a la izquierda
            b=(a+b)/2
            res=[res[0], estimate_gamma_svm([(a+b)/2], p, c)[0], res[1]]
        else: # en otro caso nos vamos a la derecha
            a=(a+b)/2
            res=[res[1], estimate_gamma_svm([(a+b)/2], p, c)[0], res[2]]
        #print(res)
        # vamos guardando accuracies
        track_g.append([(a+b)/2, res[1]])
        # si estoy ante un nuevo máximo accuracy lo guardo junto con valor gamma
        if np.max(res) > b_r[1]:
            b_r=[[a,(a+b)/2,b][np.argmax(res)], np.max(res)]
    track_g_m.append(track_g) # guardo progreso para cada c
    best_results_m.append(b_r) # voy guardando mejores accuracies para cada c


index_best_acc = np.argmax(np.array(best_results_m)[:,1])
w_p_c = c_params[index_best_acc]
w_p_gam = best_results_m[index_best_acc][0]

print("Parámetros ganadores c={}, con gamma={}".format(w_p_c, w_p_gam))

#mostramos progreso mejor accuracy
plt.plot(np.array(track_g_m[index_best_acc])[:,0], np.array(track_g_m[index_best_acc])[:,1], color='k', ls='-')
plt.title('Progresión de gamma para C={}'.format(w_p_c))
plt.xlabel('gamma')
plt.ylabel('cross validation accuracy')
plt.show()

wait()

# Entrenamiento con los parámetros ganadores
models = [{"model": [SVC(C=w_p_c, # hiperparámetro estimado c, w_p_c=10
        kernel='rbf', # kernel Radial Basis Function
        degree=3, # solo para kernel poly, no es nuestro caso
        gamma=w_p_gam, # hiperparámetro estimado gamma, w_p_gam=0.02007346725463867
        coef0=0.0, # solo para kernel poly, no es nuestro caso
        shrinking=True, # suponemos número alto de iteraciones, nos podrá ayudar a reducir tiempo de cómputo
        probability=False, # consideramos que no es necesario y nos aumentaría coste computacional
        tol=0.001, # tolerancia para criterio parada, hasta milésima
        cache_size=200, # puede mejorar tiempo ejecución para problemas de muchos datos, en nuestro caso consideramos que son pocos datos y que con 200MB sería suficiente
        class_weight=None, # suponemos todas las clases con peso 1 pues tenemos clases balanceadas
        verbose=False, # no queremos mensajes
        max_iter=-1, # no establecemos criterio de parada por iteraciones
        decision_function_shape='ovr', # devolvemos función decisión one vs rest 
        break_ties=False, # no consideramos casos de empates, y ahorraremos considerablemente en coste computacional
        random_state=SEED)], # para tener reproducibilidad
        }
]

winner_model = model_selection.GridSearchCV(p, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)

print("Vector con la media de accuracy por val. cruzada de cada modelo en orden de ejecución")
print(winner_model.cv_results_['mean_test_score'])
print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format( winner_model.score(X_test, y_test)))

# Mostramos curva de aprendizaje del modelo ganador
plot_learning_curve(winner_model, 't', X_train, y_train, cv=10, n_jobs=-1)
plt.show()

# Matriz de confusión
pr=winner_model.predict(X_test)
cm=metrics.confusion_matrix(y_test,pr)
plt.figure(figsize = (13,8))
plt.title("Matriz de confusión")
sns.set(font_scale=1)
sns.heatmap(cm, annot=True, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))

print("Calculamos accuracy de modelo aleatorio")
# Por úlitmo vemos desempeño de clasificador aleatorio siguiendo distrib uniforme 
dummy = Pipeline([("dummy", DummyClassifier(strategy="uniform", random_state=SEED))])
dummy.fit(X_train,y_train)
print("Accuracy modelo aleatorio en test: {}".format(dummy.score(X_test, y_test)))
print("Acccuracy modelo aleatorio en training: {}".format(dummy.score(X_train,y_train)))

wait()
#######################################################################
############################ SEGUNDA PARTE ############################
#######################################################################

# Es todo el código igual pero con VarianceThreshold(0) en lugar de VarianceThreshold(0.05) 
print("SEGUNDA PARTE")
print("Es todo el código igual pero con VarianceThreshold(0) en lugar de VarianceThreshold(0.05)")
wait()
# Quitamos caract con varianza 0 para poder computar la matriz de coef de correlación lineal
X_varth = Pipeline([('varth', VarianceThreshold())]).fit_transform(X_train, y_train) 
corr_matrix = np.abs( np.corrcoef(np.transpose(X_varth)) )
# Visualizamos matriz de coef. correlación mediante heatmap
ax = sns.heatmap(corr_matrix, cmap="Reds")
plt.title('Heatmap de la matriz de coeficientes de correlación\n entre las variables (en valor absoluto)')
plt.show()

# Comprobamos si tenemos caract con alto coef correlación lineal, 0.95
check_high_corr(corr_matrix, 0.95)

# Calculamos las varianzas de las características
vars = []
for i in range(X_train.shape[1]):
    vars.append(np.var(X_train[:,i]))

# Dibujamos las varianzas de cada característica
plt.plot([i+1 for i in range(X_train.shape[1])], vars, color='k', ls='-')
plt.title('Varianzas por característica')
plt.xlabel('Característica')
plt.ylabel('Varianza')
plt.show()

# Eliminaremos features con varianza < 0.05
feat_low_var = []
print("Features a eliminar")
for j in range(len(vars)):
    if vars[j]<0.05:
        print("Feature {} con varianza={} < 0.05 y media {}".format(j, vars[j], np.mean(X_train[:,j] )))
        feat_low_var.append(j)

print("Número de características original: {}, después de eliminar features: {}".format(X_train.shape[1], X_train.shape[1]-len(feat_low_var)))

for f in feat_low_var:
    print("Percentil 99 de feature {} es {}".format(f, np.percentile(X_train[:,f],99)))
    
print("Percentil 98 de feature {} es {}".format(47, np.percentile(X_train[:,f],99)))

wait()

#####################################################################################
################################### MODELO LINEAL ###################################
#####################################################################################

print("MODELO LINEAL: Regresión Logística multiclase one-vs-rest")
pipeline = Pipeline([
    ('VarThr', VarianceThreshold(threshold=0.0)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('PolFeat', preprocessing.PolynomialFeatures(degree=2, # Transformamos vectores caract. a vect. caract. con combinaciones de producto caract con grado menor o igual que 2
                                                 interaction_only=False, # pues queremos todas las combinaciones
                                                 include_bias=False)), # no añadimos columna de 1s pues el siguiente paso es normalizar,  el sesgo lo añadiremos en las funciones de nuestros algoritmos de aprendizaje.
    ('Post PolFeat StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                                 with_mean=True, # centramos los datos antes de escalar
                                                                 with_std=True)), # queremos tener varianza 1
    ("model", SGDClassifier())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

# los modelos de búsqueda serán SGDClassifier con cada valor para regularización y Perceptron con cada valor para regularización, en total 6 modelos
models = [
        {"model": [SGDClassifier(
            loss = 'log', # función de pérdida de regresión logística
            penalty = 'l2', # utilizaremos regularización l2 (reg. Ridge)
            # alpha  constante del término de regularización (probaremos distintos valores mediante 10-fold cross validation)
            fit_intercept = True, # añadimos sesgo o intercept pues nuestra matriz aún no tiene columna de 1s 
            max_iter = 180, # Número máximo de iteraciones arbitrario
            #tol  Tolerancia para criterio de parada por tolerancia (parar si loss > best_loss - tol tras n_iter_no_change épocas seguidas)
            # el criterio valora si el error no mejora en tol el mejor error hasta el momento. En nuestro caso esta condición por tolerancia se usará para learning_rate adaptativo
            shuffle = True, # mezclamos después de cada época,
            n_jobs = -1, # máxima paralelización posible en ejecución
            random_state = SEED, # para tener reproducibilidad de los resultados
            learning_rate = 'adaptive', # si no se mejora resultado por criterio tolerancia (loss > best_loss - tol) tras n_iter_no_change épocas seguidas, entonces cambiamos learning rate por (learning rate)/5
                                        # queremos evitar oscilación
            eta0 = 0.05, # learning rate inicial arbitrario
            early_stopping = False, # False pues no queremos reservar más datos para validación, consideramos nuestro dataset pequeño y que no conviene quitar más datos de training
            n_iter_no_change = 5, # cada 5 épocas sin mejora en crit. tolerancia se realizará la adaptación del learning rate (learning_rate='adaptive')
            class_weight = None, # se interpreta que todas las clases tienen peso 1, se elige así pues nuestro problema tiene clases balanceadas
            average = False, # no nos interesa obtener media de pesos
            verbose = 0, # no nos interesan mensajes 
            warm_start = False, # no reutilizamos ninguna solución anterior durante la validación cruzada
            l1_ratio = 0 # 0 corresponde a l2 penalty, no se usará pues solo se usa si learning_rate = 'elasticnet'
            )], # máxima paralelización posible
         "model__alpha": [0.000001, 0.0001, 0.001], # probamos valores de regularización (hiperparámetro)
         "model__tol": [0.0001, 0.001, 0.01]} # probamos distintos valores de tolerancia (hiperparámetro)
        ]

print("Buscaremos la mejor hipótesis mediante 10-fold cross validation")

winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)


print("Vector con la media de accuracy por val. cruzada de cada modelo en orden de ejecución")
print(winner_model.cv_results_['mean_test_score'])

print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))

print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))

print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format( winner_model.score(X_test, y_test)))


wait()

######################################################################################
################################ PERCEPTRON MULTICAPA ################################
######################################################################################
print("Perceptron Multicapa")

# Vamos a buscar parámetro hidden_layer_sizes por validación cruzada

pipeline = Pipeline([
    ('VarThr', VarianceThreshold(threshold = 0.0)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                    with_mean=True, # centramos los datos antes de escalar
                                                    with_std = True)), # queremos tener varianza 1
    ("model", MLPClassifier())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

h_l = [(i,i) for i in range(50,105,5)]
estimate_hidden_layer_sizes(h_l, pipeline)

h_l=[(i,i) for i in range(90,101,1)]
estimate_hidden_layer_sizes(h_l, pipeline)

h_l = (98,98) #hiperparámetro ganador
models = [{"model": [MLPClassifier(
        hidden_layer_sizes=h_l, # hiperparámetro estimado por validación cruzada
        activation='relu', # elegimos función activación relu por ventajas ante tanh y sigmoide
        solver='adam', # elegimos método adam, robusto y eficiente, no requiere afinar tanto parámetros como sgd
        #alpha parámetro reg. l2 a estimar
        batch_size=128, # elegimos tamaño de batch arbitrario, pensamos que es un tamaño razonable para el tamaño de nuestro dataset aunque no podemos saber cómo se comportará si no es experimentalmente
        # es candidato a hiperparámetro a estimar por cv, nosotros fijamos uno por no hacer cv con mucha carga
        learning_rate='adaptive', # solo usado cuando solver='sgd', no es nuestro caso
        learning_rate_init=0.001, # elegimos learning rate inicial recomendado (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        power_t=0.5, # solo usado cuando solver='sgd', no es nuestro caso
        max_iter=250,  # número máximo de épocas arbitrario pero que muestra convergencia
        shuffle=True,  # mezclamos después de cada época
        random_state=SEED, # para tener reproducibilidad de los resultados
        tol=0.001, # tolerancia de parada, si no mejoramos una milésima tras n_iter_no_change épocas entonces paramos, pensamos que es un valor razonable
        verbose=False, # no queremos mensajes
        warm_start=False, # no reutilizamos ninguna solución anterior durante la validación cruzada
        momentum=0.9, # solo usado cuando solver='sgd', no es nuestro caso
        nesterovs_momentum=True, # solo usado cuando solver='sgd', no es nuestro caso
        early_stopping=False, # pues consideramos que tenemos un bajo tamaño de entrenamiento
        validation_fraction=0.1, # no nos importa valor pues no usamos early_stopping
        beta_1=0.9, # elegimos tasa exponential decay para primer momento recomendada (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        beta_2=0.999, # elegimos tasa exponential decay para segundo momento recomendada (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        epsilon=1e-08, # valor numerical stability para adam recomendado (pág. 2 https://arxiv.org/pdf/1412.6980.pdf) (buenos resultados por defecto)
        n_iter_no_change=10, # damos hasta 10 épocas para mejorar error en 0.001, si no se mejora pararemos
        max_fun=15000)], # solo usado cuando solver='lbfgs', no es nuestro caso
        "model__alpha": [0.00001,0.0001,0.001]} # parámetro de reg. l2 para cross validation
    ]

print("Buscamos mejor modelo mediante 10-fold cross validation")
winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )


print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)


print("Vector con la media de accuracy por val. cruzada de cada modelo")
print(winner_model.cv_results_['mean_test_score'])

print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))

print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))

print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format( winner_model.score(X_test, y_test)))


wait()

######################################################################################
################################### RANDOM FOREST ####################################
######################################################################################

print("Random Forest")

# Vamos a buscar parámetro model__hidden_layer_sizes por validación cruzada


p = Pipeline([
    ('VarThr', VarianceThreshold(threshold=0.0)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('Post PolFeat StandardScaler', preprocessing.StandardScaler(copy=True,  # no nos importa que se haga copia
                                                                 with_mean=True,  # centramos los datos antes de escalar
                                                                 with_std=True)),  # queremos tener varianza 1
    ("model",
     RandomForestClassifier())])  # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn


n_est = [i for i in range(50, 501, 50)]

estimate_n_estimators(n_est, p)     # 450

n_est = [i for i in range(400, 501, 25)]

estimate_n_estimators(n_est, p)    # 450

n_est = [i for i in range(475, 501, 5)]

estimate_n_estimators(n_est, p)     # 455

n_est = [i for i in range(475, 490, 1)]

estimate_n_estimators(n_est, p)     # 480

n_est = 480

models = [{"model": [RandomForestClassifier(
        n_estimators=n_est,  # usamos el número de estimadores calculados
        criterion='gini',   # usamos gini porque es mas rapido y nos da resultados similares
        max_depth=None,     # para que explore la profundidad maxima del arbol
        min_samples_split=2,  # número mínimo de muestras para dividir un nodo, con 2 divide siempre que no sean hojas
        min_samples_leaf=1,     # una hoja se considera cuando solo es una muestra
        min_weight_fraction_leaf=0.0,   # para que todas las hojas tenga el mismo peso
        max_features='auto',    # utiliza la raiz cuadrada del número de características
        max_leaf_nodes=None,    # para que utilice todos los arboles
        min_impurity_decrease=0.0,  # para que expanda todos los nodos
        min_impurity_split=None,    # para que expanda todos los nodos
        bootstrap=True,     # como hemos visto en teoría, mejoramos los resultados usando bootstrap
        oob_score=False,     # solo queremos que use los datos de la muestra
        n_jobs=-1,      # para que utilice todos los cores del procesador
        random_state=SEED,  # para tener reproducibilidad de los resultados
        verbose=0,  # no queremos mensajes
        warm_start=False,   # para que cree el arbol de cero
        class_weight=None,  # para que todas las clases tengan el mismo peso
        #ccp_alpha grado de penalización por complejidad. Cuanto más alto más podado. Parámetro para regularizar y evitar sobreajuste. Será hiperparámetro a estimar
        max_samples=None)],     # para que utilize todos los datos
        "model__ccp_alpha": [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]} # hiperparámetro a estimar
]

print("Buscamos mejor modelo mediante 10-fold cross validation")
winner_model = model_selection.GridSearchCV(p,  # pipeline
                                            models,  # modelos
                                            scoring='accuracy',  # métrica a valorar
                                            n_jobs=-1,  # máxima paralelización posible en ejecución
                                            refit=True,
                                            # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                            cv=10,
                                            # 10-fold cross validation (estratificado, conserva proporción de clases)
                                            verbose=0,  # no mostramos toda la información posible para mayor claridad en los resultados impresos por pantalla
                                            # pre_dispatch = njobs, lanzamos n_jobs procesos
                                            # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                            # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                            )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)

print("Vector con la media de accuracy por val. cruzada de cada modelo")
print(winner_model.cv_results_['mean_test_score'])

print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))

print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))

print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format(winner_model.score(X_test, y_test)))

wait()
############################################################################
################################### SVM ####################################
############################################################################
print("SVM")

p = Pipeline([
    ('VarThr', VarianceThreshold(threshold = 0.0)), # eliminamos features con varianza <0.05, bajo poder predictivo
    ('StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                    with_mean=True, # centramos los datos antes de escalar
                                                    with_std = True)), # queremos tener varianza 1
    ("model", SVC())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn
print("Estimación hiperparámetro gamma para cada c, hacemos búsqueda dicotómica")
# Estimación hiperparámetro gamma para cada c hacemos búsqueda dicotómica
best_results_m=[] # matriz con filas de mejores resultados (gamma y accuracy) por cada param c
track_g_m =[] # matriz con seguimiento del valor de accuracy al mover gamma, una fila para cada c
c_params=np.logspace(-3,3,7) # parámetros de regularización
# para cada parámetro C
for c in c_params:
    a=0.001 # valor inferior para gamma
    b=1000 # valor superior para gamma
    res=estimate_gamma_svm([a,(a+b)/2,b], p, c) # obtengo accuracies por cv para cada valor
    track_g=[]
    track_g.append([(a+b)/2, res[1]]) # iremos guardando  progresión aumento accuracy
    b_r=[[a,(a+b)/2,b][np.argmax(res)], np.max(res)] # iremos controlando el mejor resultado, guardamos valor gamma y mejor accuracy
    # buscaremos hasta dos cifras decimales fijas y estemos aproximando milésimas
    while np.abs(res[0]-res[1])>=0.001 or np.abs(res[1]-res[2])>=0.001:
        # si hay mejor resultado por la izquierda
        if res[0]>res[2]:
            # nos vamos a la izquierda
            b=(a+b)/2
            res=[res[0], estimate_gamma_svm([(a+b)/2], p, c)[0], res[1]]
        else: # en otro caso nos vamos a la derecha
            a=(a+b)/2
            res=[res[1], estimate_gamma_svm([(a+b)/2], p, c)[0], res[2]]
        #print(res)
        # vamos guardando accuracies
        track_g.append([(a+b)/2, res[1]])
        # si estoy ante un nuevo máximo accuracy lo guardo junto con valor gamma
        if np.max(res) > b_r[1]:
            b_r=[[a,(a+b)/2,b][np.argmax(res)], np.max(res)]
    track_g_m.append(track_g) # guardo progreso para cada c
    best_results_m.append(b_r) # voy guardando mejores accuracies para cada c


index_best_acc = np.argmax(np.array(best_results_m)[:,1])
w_p_c = c_params[index_best_acc]
w_p_gam = best_results_m[index_best_acc][0]

print("Parámetros ganadores c={}, con gamma={}".format(w_p_c, w_p_gam))

#mostramos progreso mejor accuracy
plt.plot(np.array(track_g_m[index_best_acc])[:,0], np.array(track_g_m[index_best_acc])[:,1], color='k', ls='-')
plt.title('Progresión de gamma para C={}'.format(w_p_c))
plt.xlabel('gamma')
plt.ylabel('cross validation accuracy')
plt.show()

wait()

# Entrenamiento con los parámetros ganadores
models = [{"model": [SVC(C=w_p_c, # hiperparámetro estimado c, w_p_c=10
                         kernel='rbf', # kernel Radial Basis Function
                         degree=3, # solo para kernel poly, no es nuestro caso
                         gamma=w_p_gam, # hiperparámetro estimado gamma, w_p_gam=0.02007346725463867
                         coef0=0.0, # solo para kernel poly, no es nuestro caso
                         shrinking=True, # suponemos número alto de iteraciones, nos podrá ayudar a reducir tiempo de cómputo
                         probability=False, # consideramos que no es necesario y nos aumentaría coste computacional
                         tol=0.001, # tolerancia para criterio parada, hasta milésima
                         cache_size=200, # puede mejorar tiempo ejecución para problemas de muchos datos, en nuestro caso consideramos que son pocos datos y que con 200MB sería suficiente
                         class_weight=None, # suponemos todas las clases con peso 1 pues tenemos clases balanceadas
                         verbose=False, # no queremos mensajes
                         max_iter=-1, # no establecemos criterio de parada por iteraciones
                         decision_function_shape='ovr', # devolvemos función decisión one vs rest 
                         break_ties=False, # no consideramos casos de empates, y ahorraremos considerablemente en coste computacional
                         random_state=SEED)], # para tener reproducibilidad
        }
]

winner_model = model_selection.GridSearchCV(p, # pipeline
                                        models, # modelos
                                        scoring = 'accuracy', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 10, # 10-fold cross validation (estratificado, conserva proporción de clases)
                                        verbose = 0, # no nos interesan mensajes
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)

print("Vector con la media de accuracy por val. cruzada de cada modelo en orden de ejecución")
print(winner_model.cv_results_['mean_test_score'])
print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))
print("\nPromedio de accuracy en validación cruzada del modelo ganador: {:1.8f}\n".format(winner_model.best_score_))
print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("Accuracy en entrenamiento: {:1.8f}".format(winner_model.score(X_train, y_train)))
print("Accuracy en test: {:1.8f}".format( winner_model.score(X_test, y_test)))


# Matriz de confusión
pr=winner_model.predict(X_test)
cm=metrics.confusion_matrix(y_test,pr)
plt.figure(figsize = (13,8))
plt.title("Matriz de confusión")
sns.set(font_scale=1)
sns.heatmap(cm, annot=True, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))


