# -*- coding: utf-8 -*-
"""P2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NZdotiVnFf2eIfZpYzJ-xl1S1tOCAZKd

## Ejercicio 1: sobre la complejidad de H y el ruido
"""

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

#Establece un punto de parada
def esperar():
  input("Pulsa enter para continuar...")
  plt.close('all')

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

def muestra_dataset(X, y, label1, label2, title=None):
    ax = scatter_data(X[y==label1], label=str(label1), title=title)
    _ = scatter_data(X[y==label2], ax=ax, label=str(label2))
    ax.legend(loc='upper right')
    return ax

def scatter_data(data, ax=None, label=None, title=None):
    if ax==None:
        _, ax = plt.subplots()
    if title!=None:
        ax.set_title(title)
    ax.scatter(data[:,0], data[:,1], label=label)
    return ax

"""1. (1 punto) Dibujar gráficas con las nubes de puntos simuladas con las siguientes condiciones:

a) Considere N = 50, dim = 2, rango = [−50, +50] con simula_unif (N, dim, rango).

b) Considere N = 50, dim = 2 y sigma = [5, 7] con simula_gaus(N, dim, sigma).
"""

X1 = simula_unif( 50, 2, [-50,50] )
plt.plot( X1[:,0], X1[:,1], 'o' );
plt.show()

X2 = simula_gaus( 50, 2, [5,7] )
plt.plot( X2[:,0], X2[:,1], 'o' );
plt.show()

esperar()

"""2. Vamos a valorar la influencia del ruido en la selección de la complejidad de la clase
de funciones. Con ayuda de la función simula_unif (100, 2, [−50, 50]) generamos una
muestra de puntos 2D a los que vamos añadir una etiqueta usando el signo de la función
f (x, y) = y − ax − b, es decir el signo de la distancia de cada punto a la recta simulada con
simula_recta().

a) (1 punto) Dibujar un gráfico 2D donde los puntos muestren (use colores) el resultado
de su etiqueta. Dibuje también la recta usada para etiquetar. (Observe que todos los
puntos están bien clasificados respecto de la recta)
"""

X = simula_unif(100,2,[-50,50])
a, b = simula_recta([-50,50])    #Parámetros de la recta

def signo(x):
    if x >= 0:
      return 1
    return -1

def f_aux(x, y, a, b):
	  return signo(y - a*x - b)
   
def f(X, a, b):
    return np.asarray([ f_aux(x,y,a,b) for x,y in X ])

foX_sin_ruido = f(X, a, b)
ax = muestra_dataset(X,foX_sin_ruido,1,-1)

#Dibujando la línea
x = np.linspace(-50,50,100)
y = a*x+b
indices = np.logical_and(y<=50, y>=-50)    #Acotamos la línea al cuadrado [-50,50]x[-50,50]
ax.plot(x[indices], y[indices]);
plt.show()

esperar()

"""b) (0.5 puntos) Modifique de forma aleatoria un 10 % de las etiquetas positivas y otro
10 % de las negativas y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo
la gráfica anterior. (Ahora habrá puntos mal clasificados respecto de la recta)
"""

#Cambia aleatoriamente el signo de un 10% de las etiquetas
def random_sign_change(foX, perc):
    new_foX = np.copy(foX)
    to_change = np.random.choice( np.arange(len(foX)), int(len(foX)*perc) )
    new_foX[to_change] = -new_foX[to_change]
    return new_foX

foX = np.copy(foX_sin_ruido)
ind_1 = foX_sin_ruido==1
ind_menos1 = foX==-1
foX[ind_1] = random_sign_change(foX[ind_1], 0.1)
foX[ind_menos1] = random_sign_change(foX[ind_menos1], 0.1)

ax = muestra_dataset(X,foX,1,-1)
ax.plot(x[indices], y[indices]);
plt.show()

esperar()

"""c) (2.5 puntos) Supongamos ahora que las siguientes funciones definen la frontera de
clasificación de los puntos de la muestra en lugar de una recta 

- f (x, y) = (x − 10)^2 + (y − 20)^2 − 400

- f (x, y) = 0,5(x + 10)^2 + (y − 20)^2 − 400

- f (x, y) = 0,5(x − 10)^2 − (y + 20)^2 − 400

- f (x, y) = y − 20x^2 − 5x + 3

Visualizar el etiquetado generado en 2b junto con cada una de las gráficas de cada
una de las funciones. Comparar las regiones positivas y negativas de estas nuevas
funciones con las obtenidas en el caso de la recta. Argumente si estas funciones más
complejas son mejores clasificadores que la función lineal. Observe las gráficas y diga
que consecuencias extrae sobre la influencia del proceso de modificación de etiquetas
en el proceso de aprendizaje. Explicar el razonamiento.
"""

def apply_f(X, f_aux):
    return np.asarray([ f_aux(x,y) for x,y in X ])

def f1(X):
    return apply_f(X, lambda x,y: (x - 10)**2 + (y - 20)**2 - 400)

def f2(X):
    return apply_f(X, lambda x,y: 0.5*(x + 10)**2 + (y - 20)**2 - 400)

def f3(X):
    return apply_f(X, lambda x,y: 0.5*(x - 10)**2 - (y + 20)**2 - 400)

def f4(X):
    return apply_f(X, lambda x,y: y - 20*x**2 - 5*x + 3)

def error(pred, truth):
    return 1-np.sum(np.equal(pred,truth))/len(truth)
                   
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz( grid[:,[0,1]] )
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

plot_datos_cuad(X, foX, f1)

f1oX = np.sign(f1(X))
print("Error con f1:",error(f1oX, foX))

plot_datos_cuad(X, foX, f2)

f2oX = np.sign(f2(X))
print("Error con f2:",error(f2oX, foX))

plot_datos_cuad(X, foX, f3)

f3oX = np.sign(f3(X))
print("Error con f3:",error(f3oX, foX))

plot_datos_cuad(X, foX, f4)

f4oX = np.sign(f4(X))
print("Error con f4:",error(f4oX, foX))

esperar()

"""## Ejercicio 2: Modelos lineales

a) (3 puntos) Algoritmo Perceptron: Implementar la función
ajusta_PLA(datos, label, max_iter, vini)
que calcula el hiperplano solución a un problema de clasificación binaria usando el
algoritmo PLA. La entrada datos es una matriz donde cada item con su etiqueta está
representado por una fila de la matriz, label el vector de etiquetas (cada etiqueta es
un valor +1 o −1), max_iter es el número máximo de iteraciones permitidas y vini
el valor inicial del vector. La función devuelve los coeficientes del hiperplano.

1) Ejecutar el algoritmo PLA con los datos simulados en los apartados 2a de la
sección.1. Inicializar el algoritmo con: a) el vector cero y, b) con vectores de
números aleatorios en [0, 1] (10 veces). Anotar el número medio de iteraciones
necesarias en ambos para converger. Valorar el resultado relacionando el punto
de inicio con el número de iteraciones.
"""

#Clase para bloquear los prints de la función. Obtenida de https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
import sys, os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#Por "convergencia", estamos entendiendo que se alcanza un error de 0 (se ha separado perfectamente la muestra), tal como se usa en teoría.
def ajusta_PLA(datos, label, max_iter, vini):
    iters = 1
    continuar = True    #Se pondrá a false si se excede el máximo de iteraciones o todos los elementos están bien clasificados
    w = np.copy(vini)
    X = np.hstack( (np.ones(datos.shape[0]).T[:,None], datos) )
    while continuar:
        errores_clasif = False
        for x, y in zip(X,label):
            if not np.array_equal( signo(w.T@x), y ):
                w += y*x
                errores_clasif = True
            iters += 1
            if iters >= max_iter:
                continuar = False
                print("Máximo número de iteraciones alcanzado.")
                break
        if not errores_clasif:
            print("Paso completo sin errores de clasificación. Convergencia alcanzada en {} iteraciones.".format(iters))
            continuar = False
    return w, iters

w_ini_0 = np.zeros(X.shape[1]+1)
w_ini_random = np.random.random( (10, X.shape[1]+1) )    #Generamos unos pesos iniciales distintos para cada experimento

_, iters_0 = ajusta_PLA( X, foX_sin_ruido, 100000, w_ini_0 )

print("Con pesos iniciales nulos, el número de iteraciones necesarias para converger es de {}.".format(iters_0))

with HiddenPrints():
    w_rand_iters = []
    for w_r in w_ini_random:
        _, iters = ajusta_PLA( X, foX_sin_ruido, 100000, w_r )
        w_rand_iters.append(iters)

    w_rand_iters = np.asarray(w_rand_iters)
    media_iters = w_rand_iters.mean()
    std_iters = w_rand_iters.std()

print("La media de iteraciones necesarias para converger usando pesos iniciales aleatorios es de {} iteraciones, con una desviación típica de {}.".format(int(media_iters), std_iters))

esperar()

"""2) Hacer lo mismo que antes usando ahora los datos del apartado 2b de la sección.1.
¿Observa algún comportamiento diferente? En caso afirmativo diga cual y las
razones para que ello ocurra.
"""

w, iters_0 = ajusta_PLA( X, foX, 100000, w_ini_0 )
descrip = np.hstack((np.ones(X.shape[0]).T[:,None], X))@w
preds = [ signo(d) for d in descrip ]
error_w_ini_0 = error(preds, foX)
print("El error de clasificación obtenido usando pesos iniciales nulos es {}.".format(error_w_ini_0))

with HiddenPrints():
    w_rand_iters = []
    total_error_classif = 0
    for w_r in w_ini_random:
        w, iters = ajusta_PLA( X, foX, 100000, w_r )
        w_rand_iters.append(iters)
        descrip = np.hstack((np.ones(X.shape[0]).T[:,None], X))@w
        preds = [ signo(d) for d in descrip ]
        total_error_classif += error(preds, foX)

    w_rand_iters = np.asarray(w_rand_iters)
    media_iters = w_rand_iters.mean()
    std_iters = w_rand_iters.std()

print("La media de iteraciones necesarias para converger usando pesos iniciales aleatorios es de {} iteraciones, con una desviación típica de {}.".format(int(media_iters), std_iters))
print("El error de clasificación promedio es {}.".format(total_error_classif/10))

esperar()

"""En ambos casos encontramos la diferencia de que se alcanza el máximo de iteraciones ($10^6$). Esto podría no ser concluyente, puesto que sabemos que el algoritmo PLA requiere de muchas iteraciones para alcanzar la convergencia, pero encaja con el conocimiento que tenemos del dataset, puesto que hemos añadido ruido aleatorio que hace que el conjunto de datos no sea separable.

b) (4 puntos) Regresión Logística: En este ejercicio crearemos nuestra propia función
objetivo f (una probabilidad en este caso) y nuestro conjunto de datos D para ver cómo
funciona regresión logística. Supondremos por simplicidad que f es una probabilidad
con valores 0/1 y por tanto que la etiqueta y es una función determinista de x.
Consideremos d = 2 para que los datos sean visualizables, y sea X = [0, 2] × [0, 2] con
probabilidad uniforme de elegir cada x ∈ X . Elegir una línea en el plano que pase por
X como la frontera entre f (x) = 1 (donde y toma valores +1) y f (x) = 0 (donde y
toma valores −1), para ello seleccionar dos puntos aleatorios de X y calcular la línea
que pasa por ambos.
"""

#def generar_recta():
#    (x11,x12), (x21,x22) = simula_unif(2, 2, [0,2])
#    a = (x22-x12)/(x21-x11)
#    b = x12-x11*a
#    return a, b

"""EXPERIMENTO: Seleccione N = 100 puntos aleatorios {$x_n$} de X y evalúe las
respuestas {$y_n$} de todos ellos respecto de la frontera elegida. Ejecute Regresión
Logística (ver condiciones más abajo) para encontrar la función solución g y evalúe el
error $E_{out}$ usando para ello una nueva muestra grande de datos (> 999). Repita el
experimento 100 veces, y
- Calcule el valor de $E_{out}$ para el tamaño nuestral N=100.
- Calcule cúantas épocas tarda en converger en promedio RL para N=100 en las
condiciones fijadas para su implementación.

Implementar Regresión Logística(RL) con Gradiente Descendente Estocástico (SGD) bajo las siguientes condiciones:
- Inicializar el vector de pesos con valores 0.
- Parar el algoritmo cuando ||w (t−1) − w (t) || < 0,01, donde w (t) denota el vector
de pesos al final de la época t. Una época es un pase completo a través de los N
datos.
- Aplicar una permutación aleatoria de {1, 2, ..N } a los índices de los datos, antes
de usarlos en cada época del algoritmo.
- Usar una tasa de aprendizaje η = 0,01
"""

from math import exp 

#ERM asociado a la regresión logística
def error_rl(X,y,w):
    return np.mean( np.log( 1+np.exp(-y*(X@w)) ) )

#Se ha usado un tamaño de minibatch de 1, como se recomienda en las diapositivas
def sgd_rl(datos, labels, vini, lr, tol=0.01, norma='infinito'):
    epochs = 0
    continuar = True    #Se pondrá a false si se detecta una variación muy pequeña en los pesos
    w = np.copy(vini)
    X = np.hstack( (np.ones(datos.shape[0]).T[:,None], datos) )
    while continuar:
        indices = np.random.permutation(len(datos))
        w_prev_epoch = np.copy(w)
        for x, y in zip(X[indices,:],labels[indices]):
            w += lr*y*x/(1+exp(y*w.T@x)) 
        
        if norma=='infinito':
            continuar = not np.allclose(w_prev_epoch, w, rtol=0.0, atol=tol)     #Consideramos la norma del máximo de la diferencia
        elif norma=='2':
            continuar = not np.linalg.norm(w_prev_epoch-w)<0.01      #Consideramos la norma 2 de la diferencia
        
        epochs += 1
    return w, epochs

def experimento(reps, norma='infinito'):
    total_erm = 0.0
    total_error_classif = 0.0
    total_erm_test = 0.0
    total_error_classif_test = 0.0
    total_epochs = 0
    for _ in range(reps):
        X = simula_unif(100, 2, [0,2])
        a, b = simula_recta([0,2])
        foX = f(X, a, b)

        w, epochs = sgd_rl(X, foX, vini=np.zeros(X.shape[1]+1), lr=0.01, norma=norma)
        total_epochs += epochs

        total_erm += error_rl( np.hstack( (np.ones(X.shape[0]).T[:,None], X) ), foX, w )

        #Etiquetamos los elementos del dataset a partir de los pesos obtenidos mediante SGD
        descrip = np.hstack((np.ones(X.shape[0]).T[:,None], X))@w
        preds = [ signo(d) for d in descrip ]

        total_error_classif += error(preds, foX)

        #Cálculo de error en test
        X_test = simula_unif(1000, 2, [0,2])
        foX_test = f(X_test, a, b)

        total_erm_test += error_rl( np.hstack( (np.ones(X_test.shape[0]).T[:,None], X_test) ), foX_test, w )

        descrip_test = np.hstack((np.ones(X_test.shape[0]).T[:,None], X_test))@w
        preds_test = [ signo(d) for d in descrip_test ]

        total_error_classif_test += error(preds_test, foX_test)
    return total_erm/reps, total_error_classif/reps, total_erm_test/reps, total_error_classif_test/reps, total_epochs/reps

avg_erm, avg_error_classif, avg_erm_test, avg_error_classif_test, avg_epochs = experimento(reps=100, norma='infinito')

print("Resultados con norma infinito:")
print("ERM medio:",avg_erm)
print("Error de clasificación medio:",avg_error_classif)
print("ERM medio en test:",avg_erm_test)
print("Error de clasificación medio en test:",avg_error_classif_test)
print("Número medio de épocas:",avg_epochs)

esperar()

avg_erm, avg_error_classif, avg_erm_test, avg_error_classif_test, avg_epochs = experimento(reps=100, norma='2')

print("Resultados con norma 2:")
print("ERM medio:",avg_erm)
print("Error de clasificación medio:",avg_error_classif)
print("ERM medio en test:",avg_erm_test)
print("Error de clasificación medio en test:",avg_error_classif_test)
print("Número medio de épocas:",avg_epochs)

esperar()

for _ in range(4):
    #Repetimos el experimento para visualizar la solución
    X = simula_unif(100, 2, [0,2])
    a, b = simula_recta([0,2])
    foX = f(X, a, b)
    
    w, epochs = sgd_rl(X, foX, vini=np.zeros(X.shape[1]+1), lr=0.01, norma='2')
    
    X_test = simula_unif(1000, 2, [0,2])
    #foX_test = f(X_test, a, b)
    
    descrip_test = np.hstack((np.ones(X_test.shape[0]).T[:,None], X_test))@w
    preds_test = np.asarray([ signo(d) for d in descrip_test ])
    
    ax = muestra_dataset(X_test, preds_test, 1, -1)
    
    ##Dibujando la línea
    x = np.linspace(0,2,100)
    y = a*x+b
    indices = np.logical_and(y<=2, y>=0)    #Acotamos la línea al cuadrado [-50,50]x[-50,50]
    ax.plot(x[indices], y[indices], color='red')
    plt.show()
    
esperar()