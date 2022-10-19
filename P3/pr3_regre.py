# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: Mario Muñoz Mesa
"""

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.linear_model import Lasso, SGDRegressor, Ridge, LassoCV
import seaborn as sns # librería para visualización de datos
sns.set_style(style='white')
#sns.set_theme() # tema por defecto
#palette = sns.color_palette("bright", 11) # paleta colores bright, un color por clase

# Fijamos semilla
np.random.seed(1)

# REGRESIÓN - Superconductivity Dataset
print('REGRESIÓN - Superconductivity Dataset')

# Funcion para leer los datos
def readData(file):
    y = [] # vector de etiquetas
    X = [] # matriz de características
    
    data = np.loadtxt(file, dtype = np.double, delimiter = ",", skiprows=1)
    n_features = data.shape[1]-1
    X = data[:, 0:n_features:1] # leemos las características
    y = data[:, n_features:].astype(int) # leemos las etiquetas
    y = np.array([y[i][0] for i in range(y.size)]) # transformamos a np.array
    
    return X, y

# Leemos los datos
X, y = readData("datos/train.csv")

N, d = X.shape
print('Dataset con {} instancias de vectores de características {}-dimensionales'.format(N,d))

# Dividimos en training y test (20% de los datos para test) mezclando previamente, random_state=1 para tener reproducibilidad de resultados
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, shuffle=True, random_state = 1)
print('Número de instancias en training: {}'.format(X_train.shape[0]))
print("Proporción de training: {}".format(X_train.shape[0]/X.shape[0]))
print('Número de instancias en test: {}'.format(X_test.shape[0]))
print("Proporción de test: {}".format(X_test.shape[0]/X.shape[0]))


input("\n--- Pulsar tecla para continuar ---\n")

print('Calculamos matriz de coeficientes de correlación (en valores absolutos)')
# Calculamos matriz de coeficientes de correlación (en valores absolutos)
corr_matrix = np.abs( np.corrcoef(np.transpose(X_train)) )
# Visualizamos matriz de coef. correlación mediante heatmap
ax = sns.heatmap(corr_matrix, cmap="Reds")
plt.title('Heatmap de la matriz de coeficientes de correlación\n entre las variables (en valor absoluto)')
plt.show()

# Buscamos las características con coef correlación > 0.95
feat_to_remove =[]
# Recorremos matriz de coef correlación
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[0]):
        # tomamos solo una parte iqda de la matriz (matriz simétrica) y si tenemos coef corr > 0.95
        if i > j and corr_matrix[i][j] > 0.95: # Si tenemos coef > 0.95
            # guardamos feature en lista
            feat_to_remove.append(i)

feat_to_remove = np.unique(np.array(feat_to_remove)) # quitamos repetidos
print('Eliminaremos los características que tengan |coef correlación| > 0.95 con otras características, en concreto: ')
print(feat_to_remove)

# Eliminamos features con alta correlación lineal (parte del preprocesado manual)
for f in feat_to_remove[::-1]:
    X_train = np.delete(X_train, f, 1) # eliminamos feature f del conjunto de training
    X_test = np.delete(X_test, f, 1) # eliminamos feature f del conjunto de test (necesario para luego evaluar resultados)
    
# Calculamos la nueva matriz de coeficientes de correlación (en valores absolutos)
corr_matrix = np.abs( np.corrcoef(np.transpose(X_train)) )

# Visualizamos matriz de coef. correlación mediante heatmap
ax = sns.heatmap(corr_matrix, cmap="Reds")
plt.title('Heatmap de la matriz de coeficientes de correlación entre las\n variables (en valor absoluto) tras eliminación caract altamente correladas')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

pipeline = Pipeline([
    ('VarThr', VarianceThreshold(threshold = 0)), # eliminamos features con varianza 0, no aportan nada y quitarlos para luego normalizar
    ('Pre-pca StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                            with_mean=True, # centramos los datos antes de escalar
                                                            with_std = True)), # queremos tener varianza 1
    ('pca', PCA(0.97, # selecciona transformaciones de características que expliquen el 97% de la varianza
                copy=True, # no nos interesa sobreescribir X
                whiten=False, # no nos interesa, después vamos a hacer transformación polinómica
                svd_solver='full', # elegimos método por descomposición total en svd (descomposición en valores singulares)
                tol=0.0, # da igual el valor, se utiliza cuando svd_solver
                iterated_power='auto', # da igual pues nuestro metodo es svd_solver='full' y no 'randomized'
                random_state=None)), # da igual el valor, solo se utiliza cuando svd_solver es ‘arpack’ o ‘randomized’
    ('PolFeat', preprocessing.PolynomialFeatures(degree=2, # Transformamos vectores caract. a vect. caract. con combinaciones de producto caract con grado menor o igual que 2
                                                 interaction_only=False, # pues queremos todas las combinaciones
                                                 include_bias = False)), # no añadimos columna de 1s pues el siguiente paso es normalizar,  el sesgo lo añadiremos en las funciones de nuestros algoritmos de aprendizaje. 
    ('Post PolFeat StandardScaler', preprocessing.StandardScaler(copy=True, # no nos importa que se haga copia
                                                                 with_mean=True, # centramos los datos antes de escalar
                                                                 with_std = True)), # queremos tener varianza 1
    ("model", SGDRegressor())]) # estimador cualquiera, es un pequeño ``truco'' para tener varios modelos en GridSearchCV. Idea tomada de los enlaces:
# https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
# y 
# https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn


# los modelos de búsqueda serán SGDClassifier con cada valor para regularización y Perceptron con cada valor para regularización, en total 6 modelos
models = [
        {"model": [SGDRegressor(loss = 'squared_loss', # función de pérdida cuadrática
                                 penalty = 'l2', # utilizaremos regularización l2
                                 # alpha  constante del término de regularización (probaremos distintos valores mediante 5-fold cross validation)
                                 fit_intercept = True, # añadimos sesgo o intercept pues nuestra matriz aún no tiene columna de 1s 
                                 max_iter = 4100, # Número máximo de iteraciones arbitrario
                                 tol = 0.000001, # Tolerancia para criterio de parada por tolerancia (parar si loss > best_loss - tol tras n_iter_no_change épocas seguidas)
                                             # el criterio valora si el error es no mejora en 0.001 el mejor error hasta el momento. En nuestro caso esta condición por tolerancia se usará para learning_rate adaptativo
                                 shuffle = False, # no nos interesa introducir más ruido
                                 random_state = 1, # para tener reproducibilidad de los resultados
                                 learning_rate = 'adaptive', # si no se mejora resultado por criterio tolerancia (loss > best_loss - tol) tras n_iter_no_change épocas seguidas, entonces cambiamos learning rate por (learning rate)/5
                                                             # queremos evitar oscilación
                                 eta0 = 0.05, # learning rate inicial arbitrario, nos permitimos que sea un poco alto pues es adaptativo
                                 early_stopping = False, # False pues no queremos reservar más datos para validación
                                 n_iter_no_change = 1, # cada época sin mejora en crit. tolerancia se realizará la adaptación del learning rate (learning_rate='adaptive')
                                 average = False, # no nos interesa obtener media de pesos
                                 verbose = 0, # no nos interesan mensajes 
                                 warm_start = False, # no reutilizamos ninguna solución anterior durante la validación cruzada
                                 l1_ratio = 0 # 0 corresponde a l2 penalty, no se usará pues solo se usa si learning_rate = 'elasticnet'
                                 )],
         "model__alpha": [0.00001, 0.001, 0.1]}, # probamos valores de regularización arbitrarios dentro de los recomendados
        {"model": [Ridge(# alpha  constante del término de regularización (probaremos distintos valores mediante 5-fold cross validation)
                         fit_intercept = True, # añadimos sesgo o intercept pues nuestra matriz aún no tiene columna de 1s
                         normalize = False, # ya hemos normalizado en preprocesado
                         copy_X = True, # no nos interesa sobreescribir X
                         max_iter = None, # pues resolveremos mediante método analítico
                         # tol lo podemos dejar por defecto pues no se va a usar
                         solver = 'svd', # resolveremos de forma analítica usando descomposición en valores singulares 
                         random_state=None)], # no nos hace falta para tener reproducibilidad de los resultados pues no se va a usar solver=sag o solver=saga. Obtendremos solución analítica
         "model__alpha": [0.00001, 0.001, 0.1]}, # probamos valores de regularización arbitrarios dentro de los recomendados
        {"model": [Lasso(# alpha  constante del término de regularización (probaremos distintos valores mediante 5-fold cross validation)
                         fit_intercept = True, # añadimos sesgo o intercept pues nuestra matriz aún no tiene columna de 1s
                         normalize = False, # ya hemos normalizado en preprocesado
                         precompute = False, # tampoco tenemos demasiados datos
                         copy_X = True, # no nos interesa sobreescribir X
                         max_iter = 3000, # número máximo de iteraciones arbitrario
                         tol = 0.0001, # el por defecto, parece una tolerancia razonable. Elección arbitraria
                         warm_start = False, # no reutilizamos soluciones anteriores como inicio
                         positive = False, # no tenemos necesidad de que los coef. sean positivos
                         random_state = 1, # para tener reproducibilidad de los resultados
                         selection = 'cyclic')], # no tenemos tol > 1e^-4 para que pueda resultar interesante 'random'
         "model__alpha": [0.00001, 0.001, 0.1]}, # probamos valores de regularización arbitrarios dentro de los recomendados  
        ]

print("Buscaremos la mejor hipótesis mediante 5-fold cross validation")

winner_model = model_selection.GridSearchCV(pipeline, # pipeline
                                        models, # modelos
                                        scoring = 'neg_mean_squared_error', # métrica a valorar
                                        n_jobs = -1, # máxima paralelización posible en ejecución
                                        refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                        cv = 5, # 5-fold cross validation
                                        verbose = 0, # mostramos toda la información posible
                                        # pre_dispatch = njobs, lanzamos n_jobs procesos
                                        # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                        # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                        )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_model.fit(X_train, y_train)


print("Vector con la media de errores cuadráticos medios por val. cruzada de cada modelo en orden de ejecución (3 primeros SGDRegressor con param. reg. 0.00001, 0.0001, 0.1, los \
siguientes 3 Ridge con 0.00001, 0.0001, 0.1 como param. reg., por último Lasso de forma análoga)")
print(-winner_model.cv_results_['mean_test_score'])

print("Parámetros del modelo ganador:\n {}\n".format(winner_model.best_params_))

print("\nPromedio de ECM en validación cruzada del modelo ganador: {:1.8f}\n".format(-winner_model.best_score_))

print("Veamos los resultados del modelo ganador entrenado en toda la muestra de entrenamiento:")
print("ECM en entrenamiento: {:1.8f}".format(-winner_model.score(X_train, y_train)))
print("ECM en test: {:1.8f}".format(-winner_model.score(X_test, y_test)))


input("\n--- Pulsar tecla para continuar ---\n")

print("Calculamos Ecv mediante 20-fold cross validation para la mejor hipótesis usando todos los datos")

X, y = shuffle(X, y, random_state = 1) # Mezclamos dataset

# Ejecutamos la parte manual del preprocesado
# Eliminamos features con alta correlación lineal
for f in feat_to_remove[::-1]:
    X = np.delete(X, f, 1) # eliminamos feature f de X

winner_model.best_params_["model"]=[winner_model.best_params_["model"]]
winner_model.best_params_["model__alpha"]=[winner_model.best_params_["model__alpha"]]
winner_all_data = model_selection.GridSearchCV(pipeline, # pipeline
                                               winner_model.best_params_, # modelo ganador
                                               scoring = 'neg_mean_squared_error', # métrica a valorar
                                               n_jobs = -1, # máxima paralelización posible en ejecución
                                               refit = True, # entrenar el mejor modelo obtenido con toda la muestra de entrenamiento
                                               cv = 20, # 20-fold cross validation
                                               verbose = 0, # mostramos toda la información posible
                                               # pre_dispatch = njobs, lanzamos n_jobs procesos
                                               # error_score = np.nan , asignamos np.nan a error_score si se produce error
                                               # return_train_score = False , es costoso computacionalmente y no lo necesitamos
                                               )

print("Ejecutando GridSearchCV (tarda en ejecutar)\n")
winner_all_data.fit(X, y)

print("\n Ecv = {}".format(-winner_all_data.best_score_))

