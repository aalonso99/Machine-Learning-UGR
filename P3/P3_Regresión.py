#!/usr/bin/env python
# coding: utf-8

# # Superconductivity Data
# 
# *Alejandro Alonso Membrilla*

###### LA PRÁCTICA SE PUEDE EJECUTAR CELDA POR CELDA DESDE Spyder
###### PARA EJECUTAR UNA SOLA CELDA, PULSE Shift+Enter O EL BOTÓN
###### DE EJECUTAR CON LA FLECHA ROJA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

seed = 1
np.random.seed(seed)

def plt_deactivate_xticks():
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off


# ### Carga y separación de los datos

# In[2]:


datadir = './data/regresion/superconduct/'
raw_data = pd.read_csv( datadir+'train.csv' )


# In[3]:


print(raw_data)


# In[4]:


print(raw_data.columns)


# In[5]:


#Comprobamos si hay valores perdidos
print("¿Hay valores perdidos?")
print(raw_data.isna().any().any())


# In[6]:


raw_X = raw_data.iloc[:,:81].to_numpy()
raw_y = raw_data.iloc[:,81].to_numpy()



# In[8]:


#Separamos en train y test de forma estratificada
X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=0.2, random_state=seed)


# ### Análisis descriptivo del conjunto de entrenamiento

# In[9]:


#Comprobamos si nuestros datos siguen una distribución normal. Aplicamos un test de hipótesis tomando 
# como hipótesis nula que los atributos de nuestro dataset sigan una distribución normal.
from scipy.stats import normaltest
_,pvalues = normaltest(X_train)
print("p-values:",pvalues)


# Con un p-valor muy inferior a $0.01$, la hipótesis nula se rechaza para todos los atributos de nuestro dataset. Los datos no siguen una distribución normal. 

# In[11]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)


# Ahora vemos a estudiar la correlación entre las distintas variables. (EXPLICAR EN QUÉ CONSISTE ESTO, Y EL TEST DE PEARSON)

# In[12]:


pearson_corr_mat = np.corrcoef(X_train.T)
plt.figure(figsize=(8,8))
plt.imshow(np.abs(pearson_corr_mat))
plt.colorbar()
plt.show()

# Observamos algunos grupos de variables altamente correladas (aquellos rectángulos de color amarillo) que indican variables redundantes. 

# In[14]:


#Eliminamos variables altamente correladas
variables_correladas = set()
for i in range(len(pearson_corr_mat)):
    for j in range(i):
        if abs(pearson_corr_mat[i,j]) > 0.9:
            variables_correladas.add(i)
X_tr_red = np.delete(X_train, list(variables_correladas), axis=1)
X_test = np.delete(X_test, list(variables_correladas), axis=1)


# In[15]:


pearson_corr_mat = np.corrcoef(X_tr_red.T)
plt.figure(figsize=(8,8))
plt.imshow(np.abs(pearson_corr_mat))
plt.colorbar()
plt.show()


# In[18]:


#Es independiente de la escala y del orden de las etiquetas
from sklearn.feature_selection import mutual_info_regression
mutual_info_vec = mutual_info_regression(X_tr_red, y_train, discrete_features='auto', random_state=seed)


# In[19]:


sns.set(rc={'figure.figsize':(20,10)})
bp = sns.barplot(x=np.arange(len(mutual_info_vec)), y=mutual_info_vec, palette="muted");
bp.set(xticklabels=[]);


# In[20]:


print("Máx. info mutua:",max(mutual_info_vec))


# Todas las características guardan algo de información mutua con el vector de etiquetas.

# In[22]:


print(X_tr_red.shape)


# Ahora vamos a analizar la escala de cada atributo de nuestro dataset. Para ello visualizamos la media y la desviación típica de cada columna.

# In[25]:


medias = np.mean(X_tr_red, axis=0)
desviaciones = np.std(X_tr_red, axis=0)
sns.set(rc={'figure.figsize':(20,10)})

m = sns.barplot(x=list(range(len(medias))), y=medias, palette="muted")
m.set_yscale("log")
m.set(xticklabels=[]);


# In[26]:


s = sns.barplot(x=list(range(len(desviaciones))), y=desviaciones, palette="muted")
s.set_yscale("log")
s.set(xticklabels=[]);


# Vemos que hay una enorme diferencia en escala entre los distintos atributos, lo que nos obliga a normalizar el dataset. También haremos un análisis de los valores extremos pero para ello es conveniente hacer primero la normalización.

# ### Normalización

# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_red = scaler.fit_transform(X_tr_red)
#A la hora de usar nuestro modelo sobre el conjunto de test necesitamos aplicar sobre este
# las mismas transformaciones que sobre el conjunto de entrenamiento
X_test = scaler.transform(X_test)


# Con los datos normalizados, resulta más fácil realizar un análisis de valores extremos.

# In[28]:


plt.boxplot( X_tr_red );
plt_deactivate_xticks();
plt.show()


# No parece haber presentes outliers flagrantes. En este caso no eliminaremos valores extremos.

# ### Selección del modelo final

# In[29]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

def plot_y(data2d, y):
    plt.scatter(data2d[:,0], data2d[:,1], c=y)

#reductor solo se usa si show_plot es True
def test_model(estimator, X, y, cv=5, show_plot=False, reductor=None):
    y_pred = cross_val_predict(estimator, X, y, cv=cv) 
    if show_plot:
        X_reduc = pca.transform(X)
        plot_y(X_reduc, y_pred)
    return mean_squared_error(y, y_pred), r2_score(y, y_pred), y_pred


# In[57]:


# Aplicamos PCA para representar las clasificaciones finales, no para reducir la dimensionalidad del dataset
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_tr_red);

sns.set(rc={'figure.figsize':(20,10)})
evr_plot = sns.barplot(
                x=list(range(len(pca.explained_variance_ratio_))), 
                y=pca.explained_variance_ratio_, 
                palette="muted")
evr_plot.set(xticklabels=[]);


# In[31]:


sns.set(rc={'figure.figsize':(11,8.5)})


# In[32]:


X_tr_pca = pca.transform(X_tr_red)
plt.scatter(X_tr_pca[:,0], X_tr_pca[:,1], c=y_train)


# In[33]:

import sklearn.linear_model as lm
from sklearn.svm import LinearSVR


# In[35]:


linReg_result = test_model(lm.LinearRegression(n_jobs=6), X_tr_red, y_train, cv=5, show_plot=True, reductor=pca)


# In[38]:


results = []
for eps in [0.0, 0.1, 0.5, 1.0]:
    for c in [1, 5, 10, 15, 20]:
        svm_model = LinearSVR(random_state=seed, max_iter=10000, dual=False, 
                              loss='squared_epsilon_insensitive', 
                              C=c, epsilon=eps)
        results.append( (eps, c, test_model(svm_model, X_tr_red, y_train, cv=5, show_plot=False)) )


# In[39]:


for eps, c, (mse, r2, y_pred) in results:
    plt.scatter(1/mse, r2, label="epsilon="+str(eps)+" + C="+str(c))
plt.legend()
plt.show()


# Vemos que ambas medidas coinciden, así que no supone ningún inconveniente coger el que tenga un valor máximo en cualquiera de las dos. 

# In[40]:


best = None
best_r2 = 0.0
for eps, c, (mse, r2, y_pred) in results:
    if r2 > best_r2:
        best_r2 = r2
        best = eps, c, (mse, r2, y_pred)

print("El modelo entrenado con regresión lineal obtiene en validación MSE={} y R2={}".format(linReg_result[0], linReg_result[1]))
print("El mejor modelo de tipo Linear SVR usa epsilon={} y C={}, y obtiene en validación MSE={} y R2={}".format(best[0], best[1], best[2][0], best[2][1]))


# In[41]:


print("Mejor resultado con Linear SVR:",best)


# In[42]:


plot_y(pca.transform(X_tr_red)[:,:2], best[2][2])


# Linear SVR obtiene resultados ligeramente mejores que regresión lineal.

# In[43]:


from sklearn.svm import SVR


# In[44]:

results_rbf = []
for eps in [0.0, 0.1, 0.5, 1.0]:
    for c in [1, 5, 10, 15, 20]:
        svm_model = SVR(max_iter=10000, 
                        C=c, epsilon=eps, 
                        kernel='rbf')
        results_rbf.append( (eps, c, test_model(svm_model, X_tr_red, y_train, cv=5, show_plot=False)) )


# In[45]:


for eps, c, (mse, r2, y_pred) in results_rbf:
    plt.scatter(1/mse, r2, label="epsilon="+str(eps)+" + C="+str(c))
plt.legend()
plt.show()


# In[46]:


rbf_best = None
rbf_best_r2 = 0.0
for eps, c, (mse, r2, y_pred) in results_rbf:
    if r2 > rbf_best_r2:
        rbf_best_r2 = r2
        rbf_best = eps, c, (mse, r2, y_pred)

print("El mejor modelo de tipo SVR con kernel RBF usa epsilon={} y C={}, y obtiene en validación MSE={} y R2={}".format(rbf_best[0], rbf_best[1], rbf_best[2][0], rbf_best[2][1]))


# In[47]:


plot_y(pca.transform(X_tr_red)[:,:2], rbf_best[2][2])


# ### Test

# Comprobaremos los resultados del mejor modelo de SVM lineal seleccionado y lo compararemos con los obtenidos por el modelo no lineal.

# In[48]:


final_lineal = LinearSVR(random_state=seed, max_iter=10000, 
                         dual=False, loss='squared_epsilon_insensitive', 
                         C=1, epsilon=0.1)

final_lineal.fit(X_tr_red, y_train)


# In[49]:


y_pred_test_lin = final_lineal.predict(X_test)


# In[50]:

print("Lineal:   ECM  |  R2")
print(mean_squared_error(y_test, y_pred_test_lin), r2_score(y_test, y_pred_test_lin))


# In[51]:


plot_y(pca.transform(X_test)[:,:2], y_pred_test_lin)


# In[52]:


final_no_lineal = SVR(max_iter=10000, 
                        C=20, epsilon=0.0, 
                        kernel='rbf')

final_no_lineal.fit(X_tr_red, y_train)


# In[53]:


y_pred_test_no_lin = final_no_lineal.predict(X_test)


# In[54]:

print("RBF:   ECM  |  R2")
print(mean_squared_error(y_test, y_pred_test_no_lin), r2_score(y_test, y_pred_test_no_lin))


# In[55]:


plot_y(pca.transform(X_test)[:,:2], y_pred_test_no_lin)


# In[56]:


### La solución buena de la buena
plot_y(pca.transform(X_test)[:,:2], y_test)

