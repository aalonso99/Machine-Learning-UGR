# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.svm import SVC
from joblib import Parallel
from joblib import delayed as delay 
from sklearn.neural_network import MLPClassifier


from scipy.stats import normaltest


#from sklearn.manifold import TSNE

seed = 1
np.random.seed(seed)

datadir = './datos/'
#Si está a True se enseñan todas las gráficas mostradas en la memoria. En caso 
# contrario solo se muestran los resultados numéricos.
show = True

def plt_deactivate_xticks():
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
#threshold es el umbral de probabilidad para pertenecer o no a la primera clase. 
# Si method no es 'predict_proba' se ignora este parámetro
def test_model(estimator, X, y, cv=5, method='predict',threshold=0.5):
    if method == 'predict':  
        y_pred = cross_val_predict(estimator, X, y, cv=cv, method=method) 
    if method == 'predict_proba':
        y_probs = cross_val_predict(estimator, X, y, cv=cv, method=method) 
        y_pred = np.empty_like(y)
        y_pred[y_probs[:,0]>threshold] = 'neg'
        y_pred[y_probs[:,0]<=threshold] = 'pos'

    return y_pred, confusion_matrix(y, y_pred)

def prec_from_conf_mat(conf_mat):
    return round( ( np.sum(conf_mat.diagonal()) )/np.sum(conf_mat), 4 )

def my_scorer(confmat):
    tn, fp, fn, tp = confmat.ravel()
    cost = 10*fp+500*fn
    return cost

def print_conf_mat(conf_mat):
    group_counts = ["{0:0.0f}".format(value) for value in
                    conf_mat.flatten()]
    norm_cm = (conf_mat.T/np.sum(conf_mat, axis=1)).T
    group_percentages = ["{0:.3%}".format(value) for value in
                         norm_cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(4,4))
    sns.heatmap(norm_cm, annot=labels, fmt='');
    plt.show();
        
'''Funciones para búsqueda de modelos'''  

#Regresión Logística    
def test_LR(solver, c, reg):
    y_pred, conf_mat = test_model(
                                LogisticRegression(
                                    random_state=seed, 
                                    C=c, dual=False, 
                                    solver=solver, penalty=reg, 
                                    max_iter=10000),
                                X_train, y_train, 
                                cv=5
                            )
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'Solver':solver, 'C':c, 'Reg':reg, 'AC':prec, 'Score':my_scorer(conf_mat)}
    

#Regresión Logística con Bagging

def test_EnsembleLR(solver, c, reg):
    y_pred, conf_mat = test_model(
                                BalancedBaggingClassifier(
                                    base_estimator=LogisticRegression(
                                        random_state=seed, 
                                        C=c, dual=False, 
                                        solver=solver, penalty=reg, 
                                        max_iter=1000), 
                                    n_estimators=300,
                                    max_samples=1500,
                                    max_features=1.0,
                                    sampling_strategy=1.0,
                                    replacement=False,
                                    random_state=seed),
                                X_train, y_train, 
                                cv=5
                            )
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'Solver':solver, 'C':c, 'Reg':reg, 'AC':prec, 'Score':my_scorer(conf_mat)}

#Bagging de SVC con kernel RBF

def test_SVC(c, gamma):
    y_pred, conf_mat = test_model(
                               BalancedBaggingClassifier(
                                      base_estimator=SVC(
                                              random_state=seed,
                                              C=c, kernel = 'rbf',
                                              gamma = gamma,
                                              max_iter=1000), 
                                      n_estimators=300,
                                      max_samples=1500,
                                      max_features=1.0,
                                      sampling_strategy=1.0,
                                      replacement=False,
                                      random_state=seed), 
                                      X_train, y_train,cv=5)
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'C':c, 'G':gamma, 'AC':prec, 'Score': my_scorer(conf_mat) }
      

#Random Forest con submuestro balanceado

def test_RF(n_estim, alpha):
    model = BalancedRandomForestClassifier(n_estimators=n_estim,
                bootstrap=True, criterion='entropy', 
                max_features='sqrt', min_impurity_decrease=0.0,
                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                oob_score=False, ccp_alpha=alpha,
                n_jobs=-1, verbose=0, random_state=seed)
    
    y_pred, conf_mat = test_model(
                                model,
                                X_train, y_train, 
                                cv=5
                            )
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'Estimators':n_estim,'Alpha':alpha,'AC':prec, 'Score':my_scorer(conf_mat)}    
    
    
# Random Forest con Clasificación Ponderada

def test_RF_filtered(n_estim, threshold):
    model = BalancedRandomForestClassifier(n_estimators=n_estim,
                bootstrap=True, criterion='entropy', 
                max_features='sqrt', min_impurity_decrease=0.0,
                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                oob_score=False, ccp_alpha=0.0,
                n_jobs=-1, verbose=0, random_state=seed)
    
    y_pred, conf_mat = test_model(
                                model,
                                X_train, y_train, 
                                cv=5,
                                method='predict_proba',
                                threshold=threshold
                            )
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'Estimators':n_estim,'Thres':threshold, 'AC':prec, 'Score':my_scorer(conf_mat)}
    
    
#Bagging de Neural Network 

def test_EnsembleNN(hidden_layer_sizes, alpha):
    NNmodel = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='lbfgs', 
                          alpha=alpha, max_iter=1000, tol=0.0001, verbose=False, warm_start=False, 
                          max_fun=15000, random_state=seed)
    
    y_pred, conf_mat = test_model(
                                BalancedBaggingClassifier(
                                    base_estimator=NNmodel, 
                                    n_estimators=300,
                                    max_samples=1500,
                                    max_features=1.0,
                                    sampling_strategy=1.0,
                                    replacement=False,
                                    random_state=seed),
                                X_train, y_train, 
                                cv=5
                            )
    prec = prec_from_conf_mat(conf_mat)
    return y_pred, conf_mat, {'Layers':hidden_layer_sizes, 'Alpha':alpha , 'AC':prec, 'Score':my_scorer(conf_mat)} 
    
    
#Cambia el tamaño de las figuras
sns.set(rc={'figure.figsize':(20,10)})


#Leemos el conjunto de entrenamiento a la vez que lo barajamos
raw_train = pd.read_csv( 
                        datadir+'aps_failure_training_set.csv', 
                        header=14, na_values="na" 
                       ).sample(frac=1, random_state=seed).reset_index(drop=True)
raw_test = pd.read_csv( datadir+'aps_failure_test_set.csv', header=14, na_values="na" )



'''PREPROCESADO DE DATOS'''

# Indicamos aquellas vaiables que tienen mas del 50% de los datos missing para deshacernos de ellas
missing = raw_train.isna().sum().div(raw_train.shape[0]).mul(100).to_frame().sort_values(by=0, ascending = False)
umbral_nan_col = 50
cols_missing = missing[missing[0]>umbral_nan_col]


if show:
    '''Mostramos el histograma con el % de missing values de cada variable'''
    print("Hay {} columnas con un {}% o más de valores perdidos.".format(len(cols_missing), umbral_nan_col))
    fig, ax = plt.subplots(figsize=(15,5))
    ax.bar(missing.index, missing.values.T[0])
    plt.xticks([])
    plt.ylabel("Percentage missing")
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')

    
    
#Guardamos las etiquetas y los datos sin las columnas eliminadas.
y_train = raw_train['class'].to_numpy()
y_test = raw_test['class'].to_numpy()

cols_to_drop = list(cols_missing.index)+['class']
X_train = raw_train.drop(cols_to_drop, axis=1).to_numpy()
X_test = raw_test.drop(cols_to_drop, axis=1).to_numpy()


#Eliminamos ahora los datos que tienen missing values en mas del 20% de sus variables
n_columns = X_train.shape[1]
umbral_nan_row = 0.2*n_columns

valid_examples = np.isnan(X_train).sum(axis=1) < umbral_nan_row
X_train = X_train[valid_examples]
y_train = y_train[valid_examples]



if show:
    '''Mostramos el histograma con el número de datos etiquetados con cada clase'''
    count_tr = Counter(y_train)
    
    plt.bar(range(len(count_tr)), list(count_tr.values()), align='center', label='Instancias en D_train')
    plt.xticks(range(len(count_tr)), list(count_tr.keys()))
    
    plt.legend()
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')

    

means = np.nanmean(X_train, axis=0)
stds = np.nanstd(X_train, axis=0)

if show:
    m = sns.barplot(x=list(range(len(means))), y=means, palette="muted")
    m.set_yscale("log")
    m.set(xticklabels=[]);
    plt.show()
    
    s = sns.barplot(x=list(range(len(stds))), y=stds, palette="muted")
    s.set_yscale("log")
    s.set(xticklabels=[]);
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')
    

#Eliminamos las columnas con std=0
valid_cols = ~(stds==0)
X_train = X_train[:,valid_cols]
X_test = X_test[:,valid_cols]
means = means[valid_cols]
stds = stds[valid_cols]

# Guardamos las posiciones con valores perdidos
nan_values = np.isnan(X_train)
nan_values_test = np.isnan(X_test)

#Arreglamos los missing values
randoms = (np.random.rand(*X_train.shape)-0.5) * 3*stds

X_train[nan_values] = 0
X_train += nan_values*(means+randoms)

randoms_test = (np.random.rand(*X_test.shape)-0.5) * 3*stds

X_test[nan_values_test] = 0
X_test += nan_values_test*(means+randoms_test)


#Normalización de los datos
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if show:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.boxplot( X_train );
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')

#Comprobamos si nuestros datos siguen una distribución normal. Aplicamos un test de hipótesis tomando 
# como hipótesis nula que los atributos de nuestro dataset sigan una distribución normal.
_,pvalues = normaltest(X_train)
pvalues


# Con un p-valor de 0, la hipótesis nula se rechaza para todos los atributos de nuestro dataset. Los datos no siguen una distribución normal. 

# Ahora vemos a estudiar la correlación entre las distintas variables. (EXPLICAR EN QUÉ CONSISTE ESTO, Y EL TEST DE PEARSON)
    

if show:
    pearson_corr_mat = np.corrcoef(X_train.T)
    plt.figure(figsize=(8,8))
    plt.imshow(np.abs(pearson_corr_mat))
    plt.title("Matriz de correlación entre variables")
    plt.colorbar()
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')



# Observamos algunos grupos de variables altamente correladas que indican variables redundantes. 






pca = PCA(0.98)
pca.fit(X_train);

if show:
    #print("\nPCA: Peso por componente\n", pca.explained_variance_ratio_)
    
    evr_plot = sns.barplot(
                    x=list(range(len(pca.explained_variance_ratio_))), 
                    y=pca.explained_variance_ratio_, 
                    palette="muted").set_title('PCA: Peso por componente')
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')


X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

if show:
    pearson_corr_mat = np.corrcoef(X_train.T)
    plt.figure(figsize=(8,8))
    plt.title("Matriz de correlación entre variables (tras PCA)")
    plt.imshow(np.abs(pearson_corr_mat))
    plt.colorbar()
    plt.show()
    input('-------------- Pulse Intro para continuar --------------')



# Las nuevas variables, obtenidas mediante PCA, son incorreladas por construcción. 

# Al contar con un dataset con un ratio instancias/atributos relativamente alto, es posible aplicar métodos no paramétricos para estimar la dependencia entre las etiquetas y cada uno de los atributos. Esto es lo próximo que vamos a probar.


#Es independiente de la escala y del orden de las etiquetas (las toma como categóricas por defecto)
mutual_info_vec = mutual_info_classif(X_train, y_train, discrete_features='auto', random_state=seed)

if show:
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    bp = sns.barplot(x=np.arange(len(mutual_info_vec)), y=mutual_info_vec, palette="muted");
    bp.set(xticklabels=[]);
    plt.show();



'''Búsqueda de mejores modelos'''

'''Regresión Logística'''


results_lr = Parallel(n_jobs=5)(
                        delay(test_LR)(solver, c, reg) 
                        for solver,reg in [('saga','l1'),('lbfgs','l2')] 
                        for c in [1,10,100,1000,10000] 
                  )

for _,_,r in results_lr:
    print("Solver ->",r['Solver'],"; C ->", r['C'],"; Reg. type ->", 
          r['Reg'], "; AC ->", r['AC'], "; Score ->", r['Score'])

best_y_pred_lr = results_lr[0][0]
best_conf_mat_lr = results_lr[0][1]

print("Precision 'neg' ->",precision_score(y_train, best_y_pred_lr, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_train, best_y_pred_lr, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_train, best_y_pred_lr, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_train, best_y_pred_lr, pos_label='pos'))


    
if show:
    print_conf_mat(best_conf_mat_lr)
    
    data2d = X_train[:,:2]
    y = best_y_pred_lr
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))   
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')
    
    
    
    
'''Regresión Logística con Bagging'''

results_elr = Parallel(n_jobs=5)(
                        delay(test_EnsembleLR)(solver, c, reg) 
                        for solver,reg in [('saga','l1'),('lbfgs','l2')] 
                        for c in [1,10,100,1000,10000]
                  )

for _,_,r in results_elr:
    print("Solver ->",r['Solver'],"; C ->", r['C'],"; Reg. type ->", 
          r['Reg'], "; AC ->", r['AC'], "; Score ->", r['Score'])

best_y_pred_elr = results_elr[9][0]
best_conf_mat_elr = results_elr[9][1]

print("Precision 'neg' ->",precision_score(y_train, best_y_pred_elr, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_train, best_y_pred_elr, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_train, best_y_pred_elr, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_train, best_y_pred_elr, pos_label='pos'))



if show:
    print_conf_mat(best_conf_mat_elr)
    
    data2d = X_train[:,:2]
    y = best_y_pred_elr
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))   
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')
    
    
'''Bagging de SVC con kernel RBF'''

results_svc = Parallel(n_jobs=5)(
                        delay(test_SVC)( c, gamma) 
                        for c in [1,10,100,1000,10000] 
                        for gamma in [0.05,0.01,0.005]
                  )

for _,_,r in results_svc:
    print(" C ->", r['C'],"; G ->", r['G'], "; AC ->", r['AC'], "; Score ->", r['Score'])
    
best_y_pred_svc = results_svc[11][0]
best_conf_mat_svc = results_svc[11][1]

print("Precision 'neg' ->",precision_score(y_train, best_y_pred_svc, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_train, best_y_pred_svc, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_train, best_y_pred_svc, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_train, best_y_pred_svc, pos_label='pos'))




if show:
    print_conf_mat(best_conf_mat_svc)
    
    data2d = X_train[:,:2]
    y = best_y_pred_svc
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))   
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')
    
    
'''Random Forest con submuestreo balanceado'''

results_rf = Parallel(n_jobs=5)(
                        delay(test_RF)(n_estim, alpha) 
                        for n_estim in [100,200,400,750,1000]
                        for alpha in [0.01,0.001,0.0001,0.0]
                  )

for _,_,r in results_rf:
    print("Estimators ->", r['Estimators'],"; α ->", r['Alpha'], "; AC ->", r['AC'], "; Score ->", r['Score'])
    

#El mejor resultado se da con 400 estimadores, y criterio de separación de entropía
best_y_pred_rf = results_rf[11][0]
best_conf_mat_rf = results_rf[11][1]

print("Precision 'neg' ->",precision_score(y_train, best_y_pred_rf, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_train, best_y_pred_rf, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_train, best_y_pred_rf, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_train, best_y_pred_rf, pos_label='pos'))



    
if show:
    print_conf_mat(best_conf_mat_rf)
    
    data2d = X_train[:,:2]
    y = best_y_pred_rf
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))   
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')
    
    
'''Random Forest con Clasificación Ponderada'''

results_rf_filt = Parallel(n_jobs=6)(
                        delay(test_RF_filtered)(n_estim, thres) 
                        for n_estim in [200,400,750]
                        for thres in [0.3,0.35,0.4,0.45]
                  )

for _,_,r in results_rf_filt:
    print("Estimators ->", r['Estimators'], "; Umbral Prob. ->", r['Thres'], "; AC ->", r['AC'], "; Score ->", r['Score'])
    
    
best_y_pred_rf_filt = results_rf_filt[9][0]
best_conf_mat_rf_filt = results_rf_filt[9][1]

print("Precision 'neg' ->",precision_score(y_train, best_y_pred_rf_filt, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_train, best_y_pred_rf_filt, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_train, best_y_pred_rf_filt, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_train, best_y_pred_rf_filt, pos_label='pos'))


    
    
if show:
    print_conf_mat(best_conf_mat_rf_filt)    
    
    data2d = X_train[:,:2]
    y = best_y_pred_rf_filt
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))   
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')
    

'''Bagging de Redes Neuronales'''

results_nn = Parallel(n_jobs=6)(
                       delay(test_EnsembleNN)(hidden_layer_sizes, alpha) 
                       for hidden_layer_sizes in [ (5,5), (10,10), (25,25), (50,50) ]
                       for alpha in [ 0.0001, 0.001, 0.01 ]
                  )


for _,_,r in results_nn:
    print("Layers ->", r['Layers'], "; α ->", r['Alpha'], "; AC ->", r['AC'], "; Score ->", r['Score'])    
    
best_y_pred_nn = results_nn[4][0]
best_conf_mat_nn = results_nn[4][1]

print("Precision 'neg' ->",precision_score(y_train, best_y_pred_nn, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_train, best_y_pred_nn, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_train, best_y_pred_nn, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_train, best_y_pred_nn, pos_label='pos'))

   
    
    
if show:
    print_conf_mat(best_conf_mat_nn) 
    
    data2d = X_train[:,:2]
    y = best_y_pred_nn
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))   
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();
    
input('-------------- Pulse Intro para continuar --------------')
    
    
'''Datos originales'''
if show:
    data2d = X_train[:,:2]
    y = y_train
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();



'''ELECCIÓN DE MEJORES MODELOS EN BASE A LOS RESULTADOS OBTENIDOS'''

bb_lr = BalancedBaggingClassifier(
            base_estimator=LogisticRegression(
                random_state=seed, 
                C=10000, dual=False, 
                solver='lbfgs', penalty='l2', 
                max_iter=1000), 
            n_estimators=300,
            max_samples=1500,
            max_features=1.0,
            sampling_strategy=1.0,
            replacement=False,
            random_state=seed)

bb_svc = BalancedBaggingClassifier(
              base_estimator=SVC(
                  random_state=seed,
                  C=1000, kernel = 'rbf',
                  gamma = 0.005, class_weight = 'balanced', 
                  max_iter=10000), 
              n_estimators=300,
              max_samples=1500,
              max_features=1.0,
              sampling_strategy=1.0,
              replacement=False,
              random_state=seed)

b_rf = BalancedRandomForestClassifier(n_estimators=750,
                bootstrap=True, criterion='entropy', 
                max_features='sqrt', min_impurity_decrease=0.0,
                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                oob_score=False, 
                n_jobs=-1, verbose=0, random_state=seed)

bb_nn = BalancedBaggingClassifier(
                base_estimator=MLPClassifier(hidden_layer_sizes=(10,10), 
                                             activation='relu', solver='lbfgs', 
                                             alpha=0.001, batch_size=32, 
                                             learning_rate_init=0.001, 
                                             momentum=0.9, nesterovs_momentum=True, 
                                             early_stopping=True, validation_fraction=0.1, 
                                             beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                             max_iter=1000, shuffle=True, tol=0.0001, 
                                             verbose=False, warm_start=False, 
                                             n_iter_no_change=4, max_fun=15000, random_state=seed), 
                n_estimators=300,
                max_samples=1500,
                max_features=1.0,
                sampling_strategy=1.0,
                replacement=False,
                random_state=seed
            )

# Ajuste
bb_lr.fit(X_train, y_train);
bb_svc.fit(X_train, y_train);
b_rf.fit(X_train, y_train);
bb_nn.fit(X_train, y_train);

'''RESULTADOS'''

bb_lr_pred = bb_lr.predict(X_test)
bb_svc_pred = bb_svc.predict(X_test)

b_rf_probs = b_rf.predict_proba(X_test)
b_rf_pred = np.empty_like(y_test)
b_rf_pred[b_rf_probs[:,0]>0.35] = 'neg'
b_rf_pred[b_rf_probs[:,0]<=0.35] = 'pos'

bb_nn_pred = bb_nn.predict(X_test)

cm_bb_lr = confusion_matrix(y_test, bb_lr_pred)
cm_bb_svc = confusion_matrix(y_test, bb_svc_pred)
cm_b_rf = confusion_matrix(y_test, b_rf_pred)
cm_bb_nn = confusion_matrix(y_test, bb_nn_pred)



# Logistic Regression
print("Resultados en test del Bagging de Regresión Logística.")
print("AC ->", prec_from_conf_mat(cm_bb_lr), "; Score ->", my_scorer(cm_bb_lr))

print("Precision 'neg' ->",precision_score(y_test, bb_lr_pred, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_test, bb_lr_pred, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_test, bb_lr_pred, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_test, bb_lr_pred, pos_label='pos'))

if show:
    print_conf_mat(cm_bb_lr)
    
    data2d = X_test[:,:2]
    y = bb_lr_pred
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')



# SVC-RBF
print("Resultados en test del Bagging de SVD-RBF.")
print("AC ->", prec_from_conf_mat(cm_bb_svc), "; Score ->", my_scorer(cm_bb_svc))

print("Precision 'neg' ->",precision_score(y_test, bb_svc_pred, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_test, bb_svc_pred, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_test, bb_svc_pred, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_test, bb_svc_pred, pos_label='pos'))


if show:
    print_conf_mat(cm_bb_svc)
    
    
    data2d = X_test[:,:2]
    y = bb_svc_pred
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')



# MLP
print("Resultados en test del Bagging de Redes Neuronales.")
print("AC ->", prec_from_conf_mat(cm_bb_nn), "; Score ->", my_scorer(cm_bb_nn))

print("Precision 'neg' ->",precision_score(y_test, bb_nn_pred, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_test, bb_nn_pred, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_test, bb_nn_pred, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_test, bb_nn_pred, pos_label='pos'))

if show:
    print_conf_mat(cm_bb_nn)
    data2d = X_test[:,:2]
    y = bb_nn_pred
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')



# Random Forest
print("Resultados en test del Random Forest con clasificación ponderada.")
print("AC ->", prec_from_conf_mat(cm_b_rf), "; Score ->", my_scorer(cm_b_rf))

print("Precision 'neg' ->",precision_score(y_test, b_rf_pred, pos_label='neg'))
print("Recall 'neg' ->",recall_score(y_test, b_rf_pred, pos_label='neg'))
print("Precision 'pos' ->",precision_score(y_test, b_rf_pred, pos_label='pos'))
print("Recall 'pos' ->",recall_score(y_test, b_rf_pred, pos_label='pos'))

if show:
    print_conf_mat(cm_b_rf)
    
    
    data2d = X_test[:,:2]
    y = b_rf_pred
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();

input('-------------- Pulse Intro para continuar --------------')

    
'''DATOS EN CONJUNTO TEST'''
print("Datos de TEST:")
if show:
    data2d = X_test[:,:2]
    y = y_test
    for i in ['neg', 'pos']:
        plt.scatter(x=data2d[y==i,0], y=data2d[y==i,1], label=str(i))
    plt.legend()
    plt.xlim(-5,+30)
    plt.ylim(-15,+20)
    plt.show();
 
