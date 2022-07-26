############ PIPELINE PRODUCCIÓN (PREDICCIÓN)
'Ejecutar código para realizar el proceso de carga, preprocesamiento, entrenamiento y testeo'
# %%
import pickle

from funciones import texto_raiz

import pandas as pd
import numpy as np

#from sklearn.feature_extraction.text import  TfidfVectorizer
#from sklearn.decomposition import PCA

#from sklearn.linear_model import LogisticRegression

import xgboost as xgb
# %%
############# Carga del registro o registros para asignar predicción #########
data = pd.read_csv(r'.\Datos\dataset_mercado_publico.csv', delimiter=';')
data = data.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion'})
data.index = data['id_producto']  # cambiamos el indice del dataframe por el id_producto
data.drop(columns=['id_producto'], inplace=True)
data.drop(columns=['Kupfer'], inplace=True)
#data = data.iloc[:5,:]

df = data.copy() # hacemos una copia sobre la cual trabajar el campo descripcion
df0 = data.copy() # hacemos una copia sobre la cual trabajar las variables categoricas
df.drop(columns=['id_licitacion','Rubro1','Rubro2','Rubro3','nombre_producto'], inplace=True)
df0.drop(columns=['id_licitacion','descripcion'], inplace=True)
# %%
############# Transformación del/los registro(s) ##############
## variables categoricas
df0 = pd.get_dummies(df0, columns=['Rubro1'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['Rubro2'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['Rubro3'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['nombre_producto'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy

fh1 = open('.\Funciones\Filtro_cat.pkl','rb')
filtro = pickle.load(fh1)
fh1.close()
#### BUSCAR FORMA PARA QUE NO ARROJE ERROR CUANDO NO ENCUENTRE LAS COLUMNAS
dfcat = df0[filtro]  #dataframe filtrado solo por las columnas

#### TEMPORAL!!!! mientras para reducir numero de campos
dfcat = dfcat.iloc[:5,:]
#### FIN TEMPORAL
# %%
## NLP limpieza y preparación texto del campo descripción
df['descripcion']= df['descripcion'].astype(str) #Convertimos la columna en string para poder trabajar con el texto
df['Descripcion raiz limpia']= df['descripcion'].apply(lambda texto: texto_raiz(texto)) #Aplicamos la función texto_raiz que nos convierte las palabras en sus raíces y las limpia
# %%
## vectorizamos
fh2 = open('.\Funciones\TFIDF.pkl','rb')
vectorizador = pickle.load(fh2)
fh2.close()
df.head()

matriz_palabras = vectorizador.transform(np.array(df['Descripcion raiz limpia'])) # creamos la bolsa de palabras
matriz_palabras = matriz_palabras.astype('float32') # cambiamos el tipo a float32 para disminuir uso de memoria
df2 = pd.DataFrame(matriz_palabras.toarray())  # el array de matriz palabras pasamos a dataframe
df2.columns = vectorizador.get_feature_names() # agregamos nombres a las columnas con las palabras del vocabulario
# %%
fh3 = open('.\Funciones\PCA.pkl','rb')
pca = pickle.load(fh3)
fh3.close()
#### TEMPORAL!!!! mientras para reducir numero de campos
df2 = df2.iloc[:5,:]
##### fin TEMPORAL
dfPCA = pca.transform(df2)
lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ]
dfPCA = pd.DataFrame(dfPCA, columns = lista_PCA)
dfPCA.index = df.iloc[:5,:].index #volvemos a asignar el indice original al nuevo dataframe
# %%
X = pd.concat([dfcat,dfPCA], axis=1)
X
# %%
############# Aplicar Modelo para entregar predicción #############
## Regresion Logistica
fh4 = open('.\Modelos\m_reglog_PCA_final.pkl','rb')
logreg_model = pickle.load(fh4)
fh4.close()

# %%
y_pred_reglog = logreg_model.predict_proba(X)
y_pred_reglog
# %%
## XGBOOST
fh5 = open('.\Modelos\m_XGBoost_PCA_vfinal.pkl','rb')
xgb_model = pickle.load(fh5)
fh5.close()
xgb_model
# %%
X_dmat = xgb.DMatrix(X)
y_pred_xgboost = xgb_model.predict(X_dmat)
y_pred_xgboost

# %%
