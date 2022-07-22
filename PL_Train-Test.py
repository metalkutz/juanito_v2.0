import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from funciones import texto_limpio,texto_raiz,metrics

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import xgboost as xgb
from xgboost import XGBClassifier

########## CARGA DATOS ###################
'carga inicial dataset desde archivo csv a un objeto panda'
data = pd.read_csv(r'.\Datos\dataset_mercado_publico.csv', delimiter=';')
data = data.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion','Kupfer':'label'})
data.index = data['id_producto']  # cambiamos el indice del dataframe por el id_producto
data.drop(columns=['id_producto'], inplace=True)

########## LIMPIEZA DATOS ###################
'limpieza de datos, nans, duplicados'
data.dropna(axis=0, inplace = True) #Si alguna fila tiene un NaN se elimina la fila 
data.drop_duplicates(keep='first', inplace=True) # elimina los registros duplicados

df = data.copy() # hacemos una copia sobre la cual trabajar el campo descripcion
df0 = data.copy() # hacemos una copia sobre la cual trabajar las variables categoricas

'se eliminan columnas que no se utilizaran'
df.drop(columns=['id_licitacion','Rubro1','Rubro2','Rubro3','nombre_producto'], inplace=True)
df0.drop(columns=['id_licitacion','descripcion'], inplace=True)

'creación de variables'
df0 = pd.get_dummies(df0, columns=['Rubro1'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['Rubro2'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['Rubro3'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['nombre_producto'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy

########### NLP sobre campo DESCRIPCION #################
df['Descripcion limpia']= df['descripcion'].astype(str) #Convertimos la columna en string para poder trabajar con el texto

#Aplicamos la función texto_limpio para limpiar las descripciones
df['Descripcion limpia'] = df['Descripcion limpia'].apply(lambda texto: texto_limpio(texto)) 

# vamos a trabajar con las palabras estemizadas/raices
df['descripcion']= df['descripcion'].astype(str)
df['Descripcion raiz limpia']= df['descripcion'].apply(lambda texto: texto_raiz(texto)) #Aplicamos la función texto_raiz que nos convierte las palabras en sus raíces y las limpia

'bag of words (vector de palabras) y escalamos con TF-IDF'
# ahora vectorizamos 
descripcion = np.array(df['Descripcion raiz limpia']) # array para armar el bag of words
np.set_printoptions(precision=2)

# forma corta TFIDF vectorizer o bolsa de palabras normalizada
vectorizador = TfidfVectorizer()
matriz_palabras = vectorizador.fit_transform(descripcion) # creamos la bolsa de palabras
matriz_palabras = matriz_palabras.astype('float32') # cambiamos el tipo a float32 para disminuir uso de memoria

############ PCA SOBRE BOLSA DE PALABRAS PARA REDUCIR DIMENSIONALIDAD ##############
df2 = pd.DataFrame(matriz_palabras.toarray())  # el array de matriz palabras pasamos a dataframe
df2.columns = vectorizador.get_feature_names() # agregamos nombres a las columnas con las palabras del vocabulario
df2.shape  #51mil registros x 19mil columnas ojo que el indice es numérico con el mismo orden que el dataframe original

pca = PCA(n_components=10000) # objeto de PCA con un máximo de 10000 componentes
pca = pca.fit(df2) # ajustamos el PCA al df2 de matriz de palabras
lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ] # generamos la lista de nombres de componentes del PCA
dfPCA = pca.transform(df2)  # aplicamos la transformación al dataframe de la matriz de palabras reduciendo la dimensionalidad
dfPCA = pd.DataFrame(dfPCA, columns = lista_PCA) # agregamos nombre de las columnas asociadas a los componentes del PCA

###################  MUESTRAS TRAIN Y TEST + REBALANCEO DEL LABEL  ###########################
fh8=open(r'\Datos\array_categorias_importantes.pkl','rb') #resultado selección categorías más importantes usando modelo catboost
filtro=pickle.load(fh8)
fh8.close()

dfcat = df0[filtro]  #dataframe filtrado solo por las columnas
dfcat = dfcat.reset_index() #df0 dataframe con variables categoricas reseteamos el indice para poder concatenar
dfcat.head()

# creamos la variables independientes
X = pd.concat([dfcat,dfPCA], axis=1) # concatenamos las matrices de variables categoricas seleccionadas con el PCA de palabras
X.index = X['id_producto']  #vuelvo a dejar el índice del producto
X.drop(columns=['id_producto'], inplace=True) # elimino la columna 

# creamos la variable dependiente
y = data[['label']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

#### rebalanceo
oversampling = SMOTE(sampling_strategy=0.30) # usamos oversampling sintético podemos elegir el nivel de oversampling con  sampling_strategy=0.80
X_train_smote, y_train_smote = oversampling.fit_resample(X_train, y_train) #Se obtienen nuevos X e y

#################### TRAIN Y TEST MODELOS REGRESIÓN LOGÍSTICA Y XGBOOST #############
'PRIMERO REGRESIÓN LOGÍSTICA'
logreg = LogisticRegression()

params = {
    'C': [1.00,0.05], #valores que tomará la Inverse of regularization strength [1.00,0.05,0.01]
    'max_iter': [500], #Maximum number of iterations taken for the solvers to converge.
    'multi_class': ['ovr'], #‘ovr’, then a binary problem is fit for each label
    'penalty': ['l2']}

scoring = ['roc_auc']
grid_solver = GridSearchCV(estimator = logreg, # model to train
                   param_grid = params,
                   scoring = scoring,
                   cv = 3,  #aplica cross validation utilizando un stratified KFold
                   n_jobs=-1,
                   refit = 'roc_auc',
                   verbose = 2)

model_result_logreg = grid_solver.fit(X_train_smote,y_train_smote)  # buscamos los mejores hiperparámetros

os.chdir(r'Modelos') ## cambiamos de directorio para guardar el modelo
fh = open('m_reglog_PCA_final','wb')  #creamos archivo pickel
pickle.dump(model_result_logreg,fh)  # guardamos modelo
fh.close() # cerramos la escritura

results_cv=model_result_logreg.cv_results_  # evaluamos la estabilidad del modelo al analizar la varianza de la métrica seleccionada (AUROC)
results=pd.DataFrame(results_cv)
results.head()

metrics(model_result_logreg.best_estimator_, X_train_smote, X_test, y_train_smote, y_test, thr=0.5) #evaluamos el modelo con las metricas de clasificación

'SEGUNDO ENTRENAMIENTO XGBOOST'
dtrain = xgb.DMatrix(X_train_smote, y_train_smote) #formato datos para libreria de XGBoost
dtest = xgb.DMatrix(X_test, y_test)

##### los mejores parámetros hasta el momento, no ejecutamos Gridsearch nuevamente dado el alto uso de recurso y tiempo necesario
params={ 'base_score': 0.5, # prediccion inicial
     #'booster': ['gbtree'],# (gbtree, gblinear, dart) default=gbtree
     'colsample_bylevel': 1,
     'colsample_bytree': 0.8, #aletoreidad en selección de columnas de cada arbol
     'learning_rate': 0.05, # [0.05,0.1,0.01]  muy relacionado con el numero de estimadores, preferible learning rate bajo 0.01 y 1000 estimadores por ejemplo
     'max_depth': 6, #[2,3,4] [4,5,6]
     #'max_leaves': [0,5], #[5,10]
     'min_child_weight': 1, # minimo numero samples por hoja
     #'missing': [np.nan], # si queremos reemplazar los missings por un numero
     'n_estimators': 500, # [100,500] 100 es valor default de numero de arboles [100,150,200,250,300,350,400]
     'n_jobs': -1, # trabajos en paralelo
     #'predictor': ['gpu_predictor'], #default=auto --- Prediction using GPU. Used when tree_method is gpu_hist. only recommended for performing prediction tasks.
     'random_state': 0, # seed para generar los folds
     'reg_alpha': 0.01, # L1 regularitacion
     'reg_lambda': 0.01, # L2 regularitacion
     'scale_pos_weight': 1,
     #'tree_method': ['gpu_hist'], #default=auto ['gpu_hist'] utiliza gpu
     'subsample': 0.9} # ratio de muestras por cada arbol 
num_round = 150

xgb_model = xgb.train(params, dtrain, num_round)

fh = open('m_XGBoost_PCA_v5.pkl','wb') #guardamos modelo en archivo pickle
pickle.dump(xgb_model,fh)
fh.close()
xgb_model.save_model("m_XGBoost_v5.json") #tambien guardamos modelo en archivo json

### metricas
y_pred_probab = xgb_model.predict(dtest)
auc = roc_auc_score(y_test, y_pred_probab)
aps = average_precision_score(y_test, y_pred_probab)
y_pred= (y_pred_probab>0.5)*1
print('Test AUROC',auc)
print('Classification report with Threshold=0.5')
print(classification_report(y_test,y_pred,target_names=['0','1']))
cm = confusion_matrix(y_test,y_pred) #Matriz
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

