#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alejandro
"""

import numpy as np
from math import exp, sin, cos, pi
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, plot
import os
np.random.seed(1)

#init_notebook_mode()

#Establece un punto de parada
def esperar():
  input("Pulsa una tecla para continuar...")
  plt.close('all')
  
###Funciones para el ejercicio 1 (GD)
def E(u,v):
    return (u**3*exp(v-2)-2*v**2*exp(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*exp(v-2)-2*v**2*exp(-u))*(3*u**2*exp(v-2)+2*v**2*exp(-u))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*exp(v-2)-2*v**2*exp(-u))*(u**3*exp(v-2)-4*v*exp(-u))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#GD
def gradient_descent(loss, gradLoss, lr=0.01, initial_point=np.array([0.0, 0.0], dtype=np.float64), max_iter=1000000, 
                     error2get=float('-inf'), decay=1.0):
    hist = [initial_point]
    prevL = loss(*initial_point)
    hist_loss = [prevL]
    n_iters = 0
    while n_iters < max_iter:
      n_iters += 1
      hist.append( hist[-1]-lr*gradLoss(*hist[-1]) )
      currLoss = loss(*hist[-1])
      hist_loss.append(currLoss)
      if currLoss < error2get:
        break;
      else:
        prevL = currLoss
        lr *= decay
    return hist, n_iters, hist_loss

def f(x,y):
    return (x + 2)**2 + 2*(y - 2)**2 + 2*sin(2*pi*x)*sin(2*pi*y)

def f_arr(x,y):
    return (x + 2)**2 + 2*(y - 2)**2 + 2*np.sin(2*pi*x)*np.sin(2*pi*y)

def dfx(x,y):
    return 2*(x+2)+4*pi*cos(2*pi*x)*sin(2*pi*y)

def dfy(x,y):
    return 4*(y-2) + 4*pi*sin(2*pi*x)*cos(2*pi*y)

def gradf(x,y):
    return np.array([dfx(x,y), dfy(x,y)])

#Función para mostrar la gráfica de una función 2D junto con los puntos seguidos 
# por el proceso de optimización con GD. Las llamadas a esta función se encuentran
# comentadas
def display_gd(loss, hist, title, x_min=-3, x_max=3, y_min=-3, y_max=3):
    import plotly.graph_objects as go
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    Z = loss(X,Y)
    
    x_traj = [ point[0] for point in hist ]
    y_traj = [ point[1] for point in hist ]
    z_traj = [ loss(point[0], point[1]) for point in hist ]
    
    fig = go.Figure(
                    data=[go.Surface(x=x, y=y, z=Z)],
                    layout=go.Layout(
                              title=go.layout.Title(text=title),
                              autosize=False,
                              scene=go.layout.Scene(
                                  xaxis_title="x",
                                  yaxis_title="y",
                                  zaxis_title="f(x,y)",
                              ),
                              width=600, height=600,
                              margin=dict(l=20, r=20, b=30, t=30)
                          )
    )
    fig.add_traces(data=[go.Scatter3d(x=x_traj, y=y_traj, z=z_traj)])

    #fig.show()
    plot(fig)
    
###Funciones para el ejercicio 2 (sobre regresión lineal)
def pseudoinverse(A):
      return np.linalg.inv(A.T@A)@A.T
  
def sgd(X, y, minibatch_size, lr=0.1, epochs=50, decay=1.0):
    data = np.copy(X)
    data = np.insert(data, 0, np.ones((len(data))), axis=1)
    w = np.random.uniform(low=0.0, high=1.0, size=data.shape[1])
    n_iters = 0
    while n_iters < epochs:
        n_iters += 1
        ind_chks = chunks(np.random.permutation(len(data)), minibatch_size)
        for indices in ind_chks:
            data_minibatch = data[indices]
            y_minibatch = y[indices]
            stoch_gradient = 2/minibatch_size*np.array([
                                  np.sum( data_minibatch[:,j]*(data_minibatch@w-y_minibatch) )
                                  for j in range(len(w))
                              ])
            w -= lr * stoch_gradient
        lr *= decay
    return w
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sign(x):
    if x >= 0:
      return 1
    return -1

def line_from_w(w):
    return [0,0.8], [-w[0]/w[2], (-w[0]-0.8*w[1])/w[2]]

def error(pred, truth):
    return 1-np.sum(np.equal(pred,truth))/len(truth)

def erm(X,y,w):
    return np.mean((X@w-y)**2)

def simula_unif(N, d, size):
	  return np.random.uniform(-size,size,(N,d))

def scatter_data(data, ax=None, label=None, title=None):
    if ax==None:
        _, ax = plt.subplots()
    if title!=None:
        ax.set_title(title)
    ax.scatter(data[:,0], data[:,1], label=label)
    return ax

def muestra_dataset(X, y, label1, label2, title=None):
    ax = scatter_data(X[y==label1], label=str(label1), title=title)
    _ = scatter_data(X[y==label2], ax=ax, label=str(label2))
    ax.legend(loc='upper right')
    plt.show()
    return ax

def muestra_preds(X, y, label1, label2, title=None):
    ax = scatter_data(X[y>0], label=str(label1), title=title)
    _ = scatter_data(X[y<0], ax=ax, label=str(label2))
    ax.legend(loc='upper right')
    return ax
 
#Cambia aleatoriamente el signo de un 10% de las etiquetas
def random_sign_change(foX, perc):
    new_foX = np.copy(foX)
    to_change = np.random.choice( np.arange(len(foX)), int(len(foX)*perc) )
    new_foX[to_change] = -new_foX[to_change]
    return new_foX

#Encapsula el experimento de repetir el muestreo uniforme de train y test, la etiquetación de
# los elementos y la clasificación mediante SGD, tantas veces como se indique 
# por el parámetro reps (para esta práctica, 1000 repeticiones)
def test_experimento(reps, f):
    f_vec = np.vectorize(f)
    sum_error_clasif_tr = 0.0
    sum_error_clasif_te = 0.0
    sum_Ein = 0.0
    sum_Eout = 0.0
    for i in range(reps):
        X = simula_unif(1000, 2, 1)
        X_test = simula_unif(1000, 2, 1)

        foX = f_vec(X[:,0], X[:,1])
        foX = random_sign_change(foX, 0.1)
        foX_te = f_vec(X_test[:,0], X_test[:,1])
        foX_te = random_sign_change(foX_te, 0.1)

        w_sgd = sgd(X, foX, 32, lr=0.01, epochs=10)

        X_tr_mat = np.insert( np.copy(X), 0, np.ones((len(X))), axis=1 )
        X_te_mat = np.insert( np.copy(X_test), 0, np.ones((len(X_test))), axis=1 )
        
        preds_sgd_trainset = X_tr_mat @ w_sgd
        preds_sgd_trainset = np.array(list(map(sign, preds_sgd_trainset)))
        preds_sgd_testset = X_te_mat @ w_sgd
        preds_sgd_testset = np.array(list(map(sign, preds_sgd_testset)))

        sum_error_clasif_tr += error(preds_sgd_trainset, foX)
        sum_error_clasif_te += error(preds_sgd_testset, foX_te)
        
        sum_Ein += erm(X_tr_mat, foX, w_sgd)
        sum_Eout += erm(X_te_mat, foX_te, w_sgd)
    
    avg_error_clasif_tr = sum_error_clasif_tr/reps
    avg_error_clasif_te = sum_error_clasif_te/reps
    avg_Ein = sum_Ein/reps
    avg_Eout = sum_Eout/reps

    return avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout

#Devuelve la lista de transformaciones no lineales que aplicar a los elementos 
# del dataset
def transformations():
    return [
              (lambda x1, x2: x1),
              (lambda x1, x2: x2),
              (lambda x1, x2: x1*x2),
              (lambda x1, x2: x1*x1),
              (lambda x1, x2: x2*x2)
           ]

#Genera un nuevo conjunto de datos a partir de X, formado por una columna de 1s
# y otra columna por cada transformación en la lista transformations
def transform_characteristics(X, transformations):
    # Aplicamos las transformaciones deseadas sobre las características originales
    trans_data = np.array([
                np.array([ f(*x_row) for f in transformations ])
                for x_row in X
    ])
    return trans_data

#Encapsula el experimento de repetir el muestreo uniforme de train y test, la etiquetación de
# los elementos y la clasificación mediante SGD, tantas veces como se indique 
# por el parámetro reps (para esta práctica, 1000 repeticiones). Añade el cálculo
# de las características no lineales
def test_NLChars(reps, f, batch=32):
    f_vec = np.vectorize(f)
    sum_error_clasif_tr = 0.0
    sum_error_clasif_te = 0.0
    sum_Ein = 0.0
    sum_Eout = 0.0
    for i in range(reps):
        X = simula_unif(1000, 2, 1)
        X_test = simula_unif(1000, 2, 1)

        foX = f_vec(X[:,0], X[:,1])
        foX = random_sign_change(foX, 0.1)
        foX_te = f_vec(X_test[:,0], X_test[:,1])
        foX_te = random_sign_change(foX_te, 0.1)

        NL_X = transform_characteristics(X, transformations())
        w_sgd = sgd(NL_X, foX, batch, lr=0.01, epochs=10)

        X_tr_mat = np.insert( np.copy(NL_X), 0, np.ones((len(X))), axis=1 )
        NL_X_test = transform_characteristics(X_test, transformations())
        X_te_mat = np.insert( np.copy(NL_X_test), 0, np.ones((len(NL_X_test))), axis=1 )
        
        preds_sgd_trainset = X_tr_mat @ w_sgd
        preds_sgd_trainset = np.array(list(map(sign, preds_sgd_trainset)))
        preds_sgd_testset = X_te_mat @ w_sgd
        preds_sgd_testset = np.array(list(map(sign, preds_sgd_testset)))

        sum_error_clasif_tr += error(preds_sgd_trainset, foX)
        sum_error_clasif_te += error(preds_sgd_testset, foX_te)
        
        sum_Ein += erm(X_tr_mat, foX, w_sgd)
        sum_Eout += erm(X_te_mat, foX_te, w_sgd)
    
    avg_error_clasif_tr = sum_error_clasif_tr/reps
    avg_error_clasif_te = sum_error_clasif_te/reps
    avg_Ein = sum_Ein/reps
    avg_Eout = sum_Eout/reps

    return avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout

###Funciones relacionadas con el BONUS sobre el método de Newton    
#Segundas derivadas parciales
def d2fxx(x,y):
    return 2 - 8*(pi**2)*sin(2*pi*x)*sin(2*pi*y)

def d2fxy(x,y):                                       # Igual a d2fyx
    return 8*(pi**2)*cos(2*pi*x)*cos(2*pi*y)

def d2fyy(x,y):
    return 4 - 8*(pi**2)*sin(2*pi*x)*sin(2*pi*y)

def hessianf(x,y):
    return np.array(
                    [[d2fxx(x,y), d2fxy(x,y)],
                     [d2fxy(x,y), d2fyy(x,y)]]
                   )
#Newton
def NR_method1(loss, gradLoss, hessianLoss, lr=0.1, initial_point=np.array([0.0, 0.0], dtype=np.float64), 
              max_iter=1000000, decay=1.0):
    hist = [initial_point]
    hist_loss = [loss(*initial_point)]
    n_iters = 0
    while n_iters < max_iter:
        n_iters += 1
        invH = np.linalg.inv(hessianLoss(*hist[-1]))
        nextP = hist[-1] - lr*invH@gradLoss(*hist[-1])
        nextLoss = loss(*nextP)
        hist.append( nextP )
        hist_loss.append( nextLoss )
        lr *= decay
    return hist, n_iters, hist_loss

#Añade el cambio de signo para corregir la convergencia a puntos críticos 
# no mínimos
def NR_method2(loss, gradLoss, hessianLoss, lr=0.1, initial_point=np.array([0.0, 0.0], dtype=np.float64), 
              max_iter=1000000, decay=1.0):
    hist = [initial_point]
    hist_loss = [loss(*initial_point)]
    n_iters = 0
    while n_iters < max_iter:
      n_iters += 1
      invH = np.linalg.inv(hessianLoss(*hist[-1]))
      s = -1 if np.linalg.det(invH) < 0 or np.linalg.det(invH) > 0 and np.trace(invH) < 0 else 1
      nextP = hist[-1] - lr*invH@gradLoss(*hist[-1])*s
      nextLoss = loss(*nextP)
      hist.append( nextP )
      hist_loss.append( nextLoss )
      lr *= decay
    return hist, n_iters, hist_loss

class P1:
    
    def __init__(self):
        self.hist_loss1 = None
        self.hist_loss2 = None
    
    def apartado11y12(self):
        hist, it, hist_loss = gradient_descent(loss=E, gradLoss=gradE, lr=0.1, initial_point = np.array([1.0,1.0]), error2get=1e-14)
    
        print ('Número de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', hist[-1][0], ', ', hist[-1][1],')')
        print ('Valor de la función objetivo: (', hist_loss[-1],')')
    
    def apartado13a(self):
        hist, it, self.hist_loss1 = gradient_descent(loss=f, gradLoss=gradf, lr=0.01, initial_point = np.array([-1.0,1.0]), max_iter=50)
    
        print('eta=0.01')
        print ('Número de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', hist[-1][0], ', ', hist[-1][1],')')
        print ('Valor de la función objetivo: (', self.hist_loss1[-1],')')
        
        #display_gd(f_arr, hist, title="Ejercicio 1.3.a: eta = 0.01", x_min=-2, x_max=0, y_min=0, y_max=2)
        
        esperar()
        
        hist, it, self.hist_loss2 = gradient_descent(loss=f, gradLoss=gradf, lr=0.1, initial_point = np.array([-1.0,1.0]), max_iter=50)
    
        print('eta=0.1')
        print ('Número de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', hist[-1][0], ', ', hist[-1][1],')')
        print ('Valor de la función objetivo: (', self.hist_loss2[-1],')')
        
        #display_gd(f_arr, hist, title="Ejercicio 1.3.a: eta = 0.1", x_min=-5, x_max=0, y_min=0, y_max=4)
        
        esperar()
        
        ###Mostramos en una gráfica la evolución de la función de pérdida a lo largo de
        ### cada experimento. Observamos que la gráfica correspondiente al GD con un lr
        ### más bajo presenta una evolución mucho más estable.
        plt.plot(np.arange( max(len(self.hist_loss1), len(self.hist_loss2)) ),
                 self.hist_loss1, label="lr = 0.01")       
        plt.plot(np.arange( max(len(self.hist_loss1), len(self.hist_loss2)) ),
                 self.hist_loss2, label="lr = 0.1")  
        plt.legend()    
        plt.show()
        
    def apartado13b(self):        
        h1, _, hl1 = gradient_descent(loss=f, gradLoss=gradf, lr=0.01, initial_point = np.array([-0.5,-0.5]), max_iter=50)
        h2, _, hl2 = gradient_descent(loss=f, gradLoss=gradf, lr=0.01, initial_point = np.array([1.0,1.0]), max_iter=50)
        h3, _, hl3 = gradient_descent(loss=f, gradLoss=gradf, lr=0.01, initial_point = np.array([2.1,-2.1]), max_iter=50)
        h4, _, hl4 = gradient_descent(loss=f, gradLoss=gradf, lr=0.01, initial_point = np.array([-3.0,3.0]), max_iter=50)
        h5, _, hl5 = gradient_descent(loss=f, gradLoss=gradf, lr=0.01, initial_point = np.array([-2.0,2.0]), max_iter=50)
        
        print ('Coordenadas (-0.5, -0.5): (%.4f, %.4f)' % (h1[-1][0], h1[-1][1]))
        print ('Función objetivo (-0.5, -0.5): %.4f' % hl1[-1])
        
        print ('Coordenadas (1.0,1.0): (%.4f, %.4f)' % (h2[-1][0], h2[-1][1]))
        print ('Función objetivo (1.0,1.0): %.4f' % hl2[-1])
        
        print ('Coordenadas (2.1,-2.1): (%.4f, %.4f)' % (h3[-1][0], h3[-1][1]))
        print ('Función objetivo (2.1,-2.1): %.4f' % hl3[-1])
        
        print ('Coordenadas (-3.0,3.0):(%.4f, %.4f)' % (h4[-1][0], h4[-1][1]))
        print ('Función objetivo (-3.0,3.0): %.4f' % hl4[-1])
        
        print ('Coordenadas (-2.0,2.0): (%.4f, %.4f)' % (h5[-1][0], h5[-1][1]))
        print ('Función objetivo (-2.0,2.0): %.4f' % hl5[-1])
        
        esperar()
        
        #Calculamos la desviación típica de las soluciones obtenidas con cada punto inicial
        solutions = np.array([h1[-1], h2[-1], h3[-1], h4[-1], h5[-1]])
        std_solutions = np.std(solutions, axis=0)
        print('Desviación en las soluciones:', std_solutions)
        
        f_solutions = np.array([hl1[-1], hl2[-1], hl3[-1], hl4[-1], hl5[-1]])
        std_f_solutions = np.std(f_solutions, axis=0)
        print('Desviación en el valor objetivo:', std_f_solutions)
        
    def apartado13(self):
        self.apartado13a()
        esperar()
        
        self.apartado13b()
        
    def apartado1(self):
        self.apartado11y12()
        esperar()
        
        self.apartado13()
        
    def apartado21(self):
        #Cargamos los datos de entrenamiento y test
        X_train = np.load('datos/X_train.npy')
        y_train = np.load('datos/y_train.npy')
        X_test = np.load('datos/X_test.npy')
        y_test = np.load('datos/y_test.npy')
        
        #Eliminamos aquellas instancias de las muestras de entrenamiento y de test 
        # que no estén etiquetadas como 1 ó 5
        X_train = X_train[np.logical_or(y_train==1, y_train==5)]
        y_train = y_train[np.logical_or(y_train==1, y_train==5)]
        X_test = X_test[np.logical_or(y_test==1, y_test==5)]
        y_test = y_test[np.logical_or(y_test==1, y_test==5)]
        
        print('Tamaño del conjunto de entrenamiento:', len(y_train))
        print('Tamaño del conjunto de test:', len(y_test))
        
        y_train_labels = np.array(list(map(lambda x: 1 if x==1 else -1, y_train)), dtype=np.float64)
        y_test_labels = np.array(list(map(lambda x: 1 if x==1 else -1, y_test)), dtype=np.float64)
        
        #Mostramos un scatter plot con las muestras de entrenamiento y test
        muestra_dataset(X_train, y_train, 1, 5, title="Conjunto de entrenamiento")
        muestra_dataset(X_test, y_test, 1, 5, title="Conjunto de test")
        esperar()
        
        #Cálculo de los pesos óptimos con la pseudoinversa
        X = np.insert( X_train, 0, np.ones((len(X_train))), axis=1 )
        w = pseudoinverse(X)@y_train_labels
        
        #X_tr_mat y X_te_mat son las matrices con los datos de entrenamiento
        # y test a las que se le añade una columna de 1s al principio
        X_tr_mat = np.insert( X_train, 0, np.ones((len(X_train))), axis=1 )
        preds_trainset = X_tr_mat @ w
        preds_trainset = np.array(list(map(sign, preds_trainset)))
        
        X_te_mat = np.insert( X_test, 0, np.ones((len(X_test))), axis=1 )
        preds = X_te_mat @ w
        preds = np.array(list(map(sign, preds)))
        
        print("Errores obtenidos con la pseudoinversa:")
        
        err_clasf_tr=error(preds_trainset, y_train_labels)
        err_clasf_te=error(preds, y_test_labels)
        
        print("Error de clasificación en train:", err_clasf_tr)
        print("Error de clasificación en test:", err_clasf_te)
        
        Ein = erm(X_tr_mat, y_train_labels, w)
        Eout = erm(X_te_mat, y_test_labels, w)
        
        print("Ein:", Ein)
        print("Eout:", Eout)
        esperar()
        
        #Plot de las predicciones
        line = line_from_w(w)
        ax_tr = muestra_preds(X_train, preds_trainset, 1, 5, "Predicciones de entrenamiento")
        ax_tr.plot(line[0], line[1], color='g')
        ax_te = muestra_preds(X_test, preds, 1, 5, "Predicciones de test")
        ax_te.plot(line[0], line[1], color='g')
        esperar()
        
        #Cálculo de los pesos óptimos con SGD
        w_sgd = sgd(X_train, y_train_labels, 32, lr=0.01, epochs=10)
        
        preds_sgd_trainset = np.insert(X_train, 0, np.ones((len(X_train))), axis=1 ) @ w_sgd
        preds_sgd_trainset = np.array(list(map(sign, preds_sgd_trainset)))
        
        preds_sgd = np.insert( X_test, 0, np.ones((len(X_test))), axis=1 ) @ w_sgd
        preds_sgd = np.array(list(map(sign, preds_sgd)))
        
        print("Errores obtenidos con SGD:")
        err_clasf_tr=error(preds_sgd_trainset, y_train_labels)
        err_clasf_te=error(preds_sgd, y_test_labels)
        
        print("Error de clasificación en train:", err_clasf_tr)
        print("Error de clasificación en test:", err_clasf_te)
        
        X_tr_mat = np.insert( np.copy(X_train), 0, np.ones((len(X_train))), axis=1 )
        X_te_mat = np.insert( np.copy(X_test), 0, np.ones((len(X_test))), axis=1 )
        Ein = erm(X_tr_mat, y_train_labels, w_sgd)
        Eout = erm(X_te_mat, y_test_labels, w_sgd)
        
        print("Ein:", Ein)
        print("Eout:", Eout)
        esperar()
        
        #Plot de las predicciones
        line_sgd = line_from_w(w_sgd)
        ax_tr = muestra_preds(X_train, preds_sgd_trainset, 1, 5, "Predicciones de entrenamiento")
        ax_tr.plot(line_sgd[0], line_sgd[1], color='g')
        ax_te = muestra_preds(X_test, preds_sgd, 1, 5, "Predicciones de test")
        ax_te.plot(line_sgd[0], line_sgd[1], color='g')
        
    def apartado22(self):
        X = simula_unif(1000, 2, 1)
        _ = scatter_data(X)
        
        def f(x1, x2):
            return sign((x1-0.2)**2 + x2**2-0.6) 
        
        f_vec = np.vectorize(f)
        foX = f_vec(X[:,0], X[:,1])
        foX = random_sign_change(foX, 0.1)
        
        ax = scatter_data(X[foX==-1], label="f = -1")
        _ = scatter_data(X[foX==1], ax, label="f = 1")
        ax.legend()
        plt.show()
        esperar()
        
        ##USO DE CARACTERÍSTICAS LINEALES
        w_sgd = sgd(X, foX, 32, lr=0.01, epochs=10)
        
        preds_sgd_trainset = np.insert( np.copy(X), 0, np.ones((len(X))), axis=1 ) @ w_sgd
        preds_sgd_trainset = np.array(list(map(sign, preds_sgd_trainset)))
        
        print("Características lineales:")
        error_clasif = error(preds_sgd_trainset, foX)
        print("Error de clasificación:", error_clasif)
        
        X_tr_mat = np.insert( np.copy(X), 0, np.ones((len(X))), axis=1 )
        Ein = erm(X_tr_mat, foX, w_sgd)
        print("Ein:", Ein)
        esperar()
        
        #Plot de las predicciones
        line_sgd = line_from_w(w_sgd)
        ax_tr = muestra_preds(X, preds_sgd_trainset, -1, 1, "Modelo ajustado")
        ax_te = muestra_preds(X, foX, -1, 1, "Modelo ajustado (etiquetas originales)")
        esperar()
        
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_experimento(reps=1000, f=f)
        print("Errores medios (ECin, ECout, Ein, Eout):", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
        esperar()
        
        X = simula_unif(1000, 2, 1)
        foX = f_vec(X[:,0], X[:,1])
        foX = random_sign_change(foX, 0.1)
        
        ##USO DE CARACTERÍSTICAS NO LINEALES
        NL_X = transform_characteristics(X, transformations())
        w_sgd = sgd(NL_X, foX, 16, lr=0.01, epochs=10)
        
        X_tr_mat = np.insert( np.copy(NL_X), 0, np.ones((len(X))), axis=1 )
        preds_sgd_trainset = X_tr_mat @ w_sgd
        preds_sgd_trainset = np.array(list(map(sign, preds_sgd_trainset)))
        
        print("Características no lineales:")
        error_clasif = error(preds_sgd_trainset, foX)
        print("Error de clasificación:", error_clasif)
        
        Ein = erm(X_tr_mat, foX, w_sgd)
        print("Ein:", Ein)
        esperar()
        
        line_sgd = line_from_w(w_sgd)
        ax_tr = muestra_preds(X, preds_sgd_trainset, -1, 1, "Modelo ajustado")
        ax_te = muestra_preds(X, foX, -1, 1, "Modelo ajustado (etiquetas originales)")
        esperar()
        
        print("Errores medios (ECin, ECout, Ein, Eout) según el tamaño de batch:")
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_NLChars(reps=1000, f=f, batch=256)
        print("256->", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
        
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_NLChars(reps=1000, f=f, batch=128)
        print("128->", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
        
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_NLChars(reps=1000, f=f, batch=64)
        print("64->", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
        
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_NLChars(reps=1000, f=f, batch=32)
        print("32->", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
    
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_NLChars(reps=1000, f=f, batch=16)
        print("16->", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
    
        avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout = test_NLChars(reps=1000, f=f, batch=8)
        print("8->", avg_error_clasif_tr, avg_error_clasif_te, avg_Ein, avg_Eout)
        
    def apartado2(self):
        self.apartado21()
        esperar()
        
        self.apartado22()
        
    def BONUS(self):
        print("Optimización de f con el método de Newton:")
        hist, it, hist_loss_NR = NR_method1(loss=f, gradLoss=gradf, hessianLoss=hessianf, 
                                           lr=0.1, initial_point = np.array([-1.0,1.0]), max_iter=50,
                                           decay=1.0)
        
        print ('Número de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', hist[-1][0], ', ', hist[-1][1],')')
        print ('Valor de la función objetivo: (', hist_loss_NR[-1],')')
        
        #display_gd(f_arr, hist, title="BONUS: método de Newton-Raphson 1", x_min=-2, x_max=0, y_min=0, y_max=2)
        esperar()
        
        #Mostramos en una gráfica la evolución de la función de pérdida a lo largo de
        # cada experimento.
        plt.plot(np.arange( max(len(hist_loss_NR), len(hist_loss_NR)) ),
                 hist_loss_NR, label="NR") 
        plt.legend()    
        plt.show()
        esperar()
        
        print("Optimización usando versión modificada de Newton:")
        hist, it, hist_loss_NR = NR_method2(loss=f, gradLoss=gradf, hessianLoss=hessianf, 
                                           lr=0.1, initial_point = np.array([-1.0,1.0]), max_iter=50,
                                           decay=1.0)
        
        print ('Número de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', hist[-1][0], ', ', hist[-1][1],')')
        print ('Valor de la función objetivo: (', hist_loss_NR[-1],')')
        
        #display_gd(f_arr, hist, title="BONUS: método de Newton-Raphson 2", x_min=-2, x_max=0, y_min=0, y_max=2)
        esperar()
        
        plt.plot(np.arange( max(len(hist_loss_NR), len(hist_loss_NR)) ),
                 hist_loss_NR, label="NR; lr = 0.1") 
        plt.plot(np.arange( max(len(self.hist_loss1), len(self.hist_loss2)) ),
                 self.hist_loss1, label="lr = 0.01")       
        plt.plot(np.arange( max(len(self.hist_loss1), len(self.hist_loss2)) ),
                 self.hist_loss2, label="lr = 0.1")  
        plt.legend()    
        plt.show()
        esperar()
        
        print("Prueba con distintos puntos iniciales:")
        h1, _, hl1 = NR_method2(loss=f, gradLoss=gradf, hessianLoss=hessianf, lr=0.1, initial_point = np.array([-0.5,-0.5]), max_iter=50)
        h2, _, hl2 = NR_method2(loss=f, gradLoss=gradf, hessianLoss=hessianf, lr=0.1, initial_point = np.array([1.0,1.0]), max_iter=50)
        h3, _, hl3 = NR_method2(loss=f, gradLoss=gradf, hessianLoss=hessianf, lr=0.1, initial_point = np.array([2.1,-2.1]), max_iter=50)
        h4, _, hl4 = NR_method2(loss=f, gradLoss=gradf, hessianLoss=hessianf, lr=0.1, initial_point = np.array([-3.0,3.0]), max_iter=50)
        h5, _, hl5 = NR_method2(loss=f, gradLoss=gradf, hessianLoss=hessianf, lr=0.1, initial_point = np.array([-2.0,2.0]), max_iter=50)
        
        print ('Coordenadas (-0.5, -0.5): (%.4f, %.4f)' % (h1[-1][0], h1[-1][1]))
        print ('Función objetivo (-0.5, -0.5): %.4f' % hl1[-1])
        
        print ('Coordenadas (1.0,1.0): (%.4f, %.4f)' % (h2[-1][0], h2[-1][1]))
        print ('Función objetivo (1.0,1.0): %.4f' % hl2[-1])
        
        print ('Coordenadas (2.1,-2.1): (%.4f, %.4f)' % (h3[-1][0], h3[-1][1]))
        print ('Función objetivo (2.1,-2.1): %.4f' % hl3[-1])
        
        print ('Coordenadas (-3.0,3.0):(%.4f, %.4f)' % (h4[-1][0], h4[-1][1]))
        print ('Función objetivo (-3.0,3.0): %.4f' % hl4[-1])
        
        print ('Coordenadas (-2.0,2.0): (%.4f, %.4f)' % (h5[-1][0], h5[-1][1]))
        print ('Función objetivo (-2.0,2.0): %.4f' % hl5[-1])
    
def main():
    p1 = P1()
    p1.apartado1()
    esperar()
    
    p1.apartado2()
    esperar()
    
    p1.BONUS()
    esperar()
    
    if os.path.exists("temp-plot.html"):
        os.remove("temp-plot.html") # one file at a time
    
if __name__ == "__main__":
    main()