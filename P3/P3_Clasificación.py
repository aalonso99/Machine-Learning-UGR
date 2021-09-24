#!/usr/bin/env python
# coding: utf-8

# # Sensorless Drive Diagnosis
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
from collections import Counter
from sklearn.model_selection import train_test_split
#from sklearn.manifold import TSNE

seed = 1
np.random.seed(seed)

def plt_deactivate_xticks():
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
sns.set(rc={'figure.figsize':(10,10)})


# ### Carga y separación de los datos

# In[2]:


datadir = './data/clasificacion/'
raw_data = pd.read_csv( datadir+'Sensorless_drive_diagnosis.txt', sep=" ", header=None)


# In[3]:


print(raw_data)


# In[4]:


#Comprobamos si hay valores perdidos
print("¿Hay valores perdidos?")
print(raw_data.isna().any().any())


# In[5]:


raw_X = raw_data.iloc[:,:48].to_numpy()
raw_y = raw_data[48].to_numpy()


# In[6]:


#Separamos en train y test de forma estratificada
X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=0.2, random_state=seed, stratify=raw_y)


# In[7]:


#Comprobamos que el número de instancias en train y test es apropiado para cada clase
count_tr = Counter(y_train)
count_te = Counter(y_test)

plt.bar(range(len(count_tr)), list(count_tr.values()), align='center', label='Instancias en D_train')
plt.xticks(range(len(count_tr)), list(count_tr.keys()))

plt.bar(range(len(count_te)), list(count_te.values()), align='center', label='Instancias en D_test')
plt.xticks(range(len(count_te)), list(count_te.keys()))

plt.legend()
plt.show()


# Además, de la gráfica anterior observamos que el problema está perfectamente balanceado.

# ### Análisis descriptivo del conjunto de entrenamiento

# In[8]:


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


# Observamos algunos grupos de variables altamente correladas que indican variables redundantes. 

# In[15]:


#Eliminamos variables altamente correladas
variables_correladas = set()
for i in range(len(pearson_corr_mat)):
    for j in range(i):
        if abs(pearson_corr_mat[i,j]) > 0.9:
            variables_correladas.add(i)
X_tr_red = np.delete(X_train, list(variables_correladas), axis=1)
X_test = np.delete(X_test, list(variables_correladas), axis=1)


# In[17]:


pearson_corr_mat = np.corrcoef(X_tr_red.T)
plt.figure(figsize=(8,8))
plt.imshow(np.abs(pearson_corr_mat))
plt.colorbar()
plt.show()


# Al contar con un dataset con un ratio instancias/atributos relativamente alto, es posible aplicar métodos no paramétricos para estimar la dependencia entre las etiquetas y cada uno de los atributos. Esto es lo próximo que vamos a probar.


# In[20]:


#Es independiente de la escala y del orden de las etiquetas (las toma como categóricas por defecto)
from sklearn.feature_selection import mutual_info_classif
mutual_info_vec = mutual_info_classif(X_tr_red, y_train, discrete_features=False, random_state=seed)


# In[21]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
bp = sns.barplot(x=np.arange(len(mutual_info_vec)), y=mutual_info_vec, palette="muted");
bp.set(xticklabels=[]);


# In[22]:

#Vemos el valor de información mutua más alto para tener una referencia a la 
# hora de eliminar valores bajos.
print(max(mutual_info_vec))


# In[23]:


important_features = mutual_info_vec>0.05
X_tr_red = X_tr_red[:,important_features]
#X_tr_red = X_train[:,important_features]
X_test = X_test[:,important_features]


# In[24]:

#Vemos el número de características finales
print(X_tr_red.shape)


# Ahora vamos a analizar la escala de cada atributo de nuestro dataset. Para ello visualizamos la media y la desviación típica de cada columna.

# In[26]:


medias = np.mean(X_tr_red, axis=0)
desviaciones = np.std(X_tr_red, axis=0)
sns.set(rc={'figure.figsize':(11.7,8.27)})

m = sns.barplot(x=list(range(len(medias))), y=medias, palette="muted")
m.set_yscale("log")
m.set(xticklabels=[]);


# In[27]:


s = sns.barplot(x=list(range(len(desviaciones))), y=desviaciones, palette="muted")
s.set_yscale("log")
s.set(xticklabels=[]);


# Vemos que hay una enorme diferencia en escala entre los distintos atributos, lo que nos obliga a normalizar el dataset. También haremos un análisis de los valores extremos pero para ello es conveniente hacer primero la normalización.

# ### Normalización

# In[28]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_red = scaler.fit_transform(X_tr_red)
#A la hora de usar nuestro modelo sobre el conjunto de test necesitamos aplicar sobre este
# las mismas transformaciones que sobre el conjunto de entrenamiento
X_test = scaler.transform(X_test)    


# Con los datos normalizados, resulta más fácil realizar un análisis de valores extremos.

# In[29]:


plt.boxplot( X_tr_red );
plt_deactivate_xticks();
plt.show()


# In[30]:


###Código adaptado de https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
def remove_outlier(X, y, tolerance=1.5):
    q1 = np.quantile(X, 0.25, axis=0)
    q3 = np.quantile(X, 0.75, axis=0)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-tolerance*iqr
    fence_high = q3+tolerance*iqr
    mask = np.logical_and(X > fence_low, X < fence_high)
    X_out = X[mask.all(axis=1),:]
    y_out = y[mask.all(axis=1)]
    return X_out, y_out

nelems = [ len(remove_outlier(X_tr_red, y_train, tolerance=i)[0]) for i in range(50) ]


# In[31]:


s = sns.barplot(x=list(range(len(nelems))), y=nelems, palette="muted")


# In[32]:


print(X_tr_red.shape)
print(remove_outlier(X_tr_red, y_train, tolerance=25)[0].shape)

plt.boxplot( remove_outlier(X_tr_red, y_train, tolerance=25)[0] );
plt_deactivate_xticks();
plt.show()


# In[33]:


#Renormalizamos después de eliminar outliers
X_tr_red, y_tr_red = remove_outlier(X_tr_red, y_train, tolerance=25)
X_tr_red = scaler.fit_transform(X_tr_red)
X_test = scaler.transform(X_test)    

plt.boxplot( X_tr_red );
plt_deactivate_xticks();
plt.show()


# ### Selección del modelo final

# In[34]:


# Aplicamos PCA para representar las clasificaciones finales, no para reducir la dimensionalidad del dataset
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_tr_red);
print("\nPCA: Peso por componente\n", pca.explained_variance_ratio_)

evr_plot = sns.barplot(
                x=list(range(len(pca.explained_variance_ratio_))), 
                y=pca.explained_variance_ratio_, 
                palette="muted")

pca_X_tr = pca.transform(X_tr_red)[:,0:2]

# tsne = TSNE(n_components=2)
# tsne.fit(X_tr_red)


# In[35]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def plot_y(data2d, y):
    #Visualizamos los clasificados en cada clase
    for i in range(1, max(y)+1):
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-10,+30)
    plt.ylim(-15,+20)
    plt.plot()

#reductor solo se usa si show_plot es True
def test_model(estimator, X, y, cv=5, show_plot=False, reductor=None):
    y_pred = cross_val_predict(estimator, X, y, cv=cv) 
    if show_plot:
        assert len(X.shape) == 2 or reductor != None
        red_data = reductor.transform(X)[:,0:2]
        plot_y(red_data, y_pred)

    return y_pred, confusion_matrix(y, y_pred)

def prec_from_conf_mat(conf_mat):
    return round( ( np.sum(conf_mat.diagonal()) )/np.sum(conf_mat), 4 )


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


# In[37]:

#Implementamos el gridsearch explícitamente para poder personalizar la salida
#Utilizamos joblib para paralelizar el cálculo y ahorrar tiempo
from joblib import Parallel, delayed


# In[38]:


def test_LR(solver, c, reg):
    y_pred, conf_mat = test_model(
                                LogisticRegression(
                                    random_state=seed, 
                                    C=c, dual=False, 
                                    solver=solver, penalty=reg, 
                                    max_iter=10000, 
                                    multi_class='multinomial'), 
                                X_tr_red, y_tr_red, 
                                cv=5
                            )
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'Solver':solver, 'C':c, 'Reg':reg, 'AC':prec}
    
  
results_lr = Parallel(n_jobs=5)(
                       delayed(test_LR)(solver, c, reg) 
                       for solver in ['saga'] 
                       for c in [0.1,0.5,1,5,10] 
                       for reg in ['l1']
                  )

for _,_,r in results_lr:
    print("Solver ->",r['Solver'],"; C ->", r['C'],"; Reg. type ->", r['Reg'], "; AC ->", r['AC'])


# In[56]:


best_y_pred_lr = results_lr[4][0]
best_conf_mat_lr = results_lr[4][1]

sns.heatmap(best_conf_mat_lr, annot=True, fmt='.0f');


# In[57]:


plot_y(pca_X_tr, best_y_pred_lr)


# In[41]:


max_prec = 0.0
best_conf_mat_lsvm = None
best_y_pred_lsvm = None
for c in [0.1,0.5,1,5,10]:
    for reg in ['l1']:
        y_pred, conf_mat = test_model(
                                    OneVsRestClassifier(
                                        LinearSVC(
                                            random_state=seed, 
                                            C=c, penalty=reg, loss='squared_hinge',
                                            dual=False, multi_class='ovr',
                                            max_iter=10000), 
                                        n_jobs=6), 
                                    X_tr_red, y_tr_red, 
                                    cv=5
                                )
        prec = prec_from_conf_mat(conf_mat)
        print("C ->", c,"; Reg. type ->", reg, "; AC ->", prec)
        if prec > max_prec:
            prec = max_prec
            best_conf_mat_lsvm = conf_mat
            best_y_pred_lsvm = y_pred


# In[42]:


sns.heatmap(best_conf_mat_lsvm, annot=True, fmt='.0f');


# In[43]:


plot_y(pca_X_tr, best_y_pred_lsvm)


# In[58]:


def barplot_comparativo(score1, score2, label1, label2, classes, score_name=''):
    x = np.arange(len(classes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, score1, width, label=label1)
    ax.bar(x + width/2, score2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(score_name)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    fig.tight_layout()

    plt.show()


# F1-Scores:

# In[59]:


f1_lr = f1_score(y_tr_red, best_y_pred_lr, labels=list(range(1,12)), average=None)
f1_lsvm = f1_score(y_tr_red, best_y_pred_lsvm, labels=list(range(1,12)), average=None)


# In[60]:


barplot_comparativo(f1_lr, f1_lsvm, 
                    'Regresión Logística','SVM lineal', 
                    classes=list(range(1,12)), score_name='F1-Score')


# Precision score:

# In[61]:


precision_lr = precision_score(y_tr_red, best_y_pred_lr, labels=list(range(1,12)), average=None)
precision_lsvm = precision_score(y_tr_red, best_y_pred_lsvm, labels=list(range(1,12)), average=None)


# In[62]:


barplot_comparativo(precision_lr, precision_lsvm, 
                    'Regresión Logística','SVM lineal', 
                    classes=list(range(1,12)), score_name='Precision Score')


# Recall score:

# In[63]:


recall_lr = recall_score(y_tr_red, best_y_pred_lr, labels=list(range(1,12)), average=None)
recall_lsvm = recall_score(y_tr_red, best_y_pred_lsvm, labels=list(range(1,12)), average=None)


# In[64]:


barplot_comparativo(recall_lr, recall_lsvm, 
                    'Regresión Logística','SVM lineal', 
                    classes=list(range(1,12)), score_name='Recall Score')


# Aunque no sea por mucho, regresión logística obtiene mejores resultados que SVM lineal para todas las medidas. Por tanto, será nuestro modelo seleccionado.

# ### Test

# In[52]:


final_model = LogisticRegression(
                    random_state=seed, 
                    C=1.0, dual=False, 
                    solver='saga', penalty='l1', 
                    max_iter=10000, 
                    multi_class='multinomial'
                )


# In[53]:


final_model.fit(X_tr_red, y_tr_red)


# In[65]:


y_pred_test = final_model.predict(X_test)


# In[66]:


conf_mat_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_mat_test, annot=True, fmt='.0f');


# In[67]:


prec_from_conf_mat(conf_mat_test)


# In[68]:


plot_y(pca.transform(X_test)[:,0:2], y_pred_test)


# In[69]:


plot_y(pca.transform(X_test)[:,0:2], y_test)

