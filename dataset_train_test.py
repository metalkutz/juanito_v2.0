# %%
####### funciones, carga de datos inicial
from Carga_dataset import *
from funciones import *

####### os, pickle
import os
import pickle

###### pandas, numpy, funciones
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import  TfidfVectorizer, TfidfTransformer, CountVectorizer

###### PCA #######
from sklearn.decomposition import PCA

######### re-balanceo de muestras ##########
from imblearn.over_sampling import SMOTE 
# %%
df.head()
# %%
####### desde archivo Carga_dataset se trae dataframe con
'carga inicial dataset desde archivo csv a un objeto panda'
'se eliminan columnas que no se utilizaran'
'limpieza de datos, nans, duplicados'
'creación de variables'
# %%
########### NLP #################
df['Descripcion limpia']= df['descripcion'].astype(str) #Convertimos la columna en string para poder trabajar con el texto

#Aplicamos la función texto_limpio para limpiar las descripciones
df['Descripcion limpia'] = df['Descripcion limpia'].apply(lambda texto: texto_limpio(texto)) 

# vamos a trabajar con las palabras estemizadas/raices
df['descripcion']= df['descripcion'].astype(str)
df['Descripcion raiz limpia']= df['descripcion'].apply(lambda texto: texto_raiz(texto)) #Aplicamos la función texto_raiz que nos convierte las palabras en sus raíces y las limpia
df.drop(columns=['descripcion','Descripcion limpia'], axis=1, inplace=True)

# %%
'bag of words (vector de palabras) y escalamos con TF-IDF'
# ahora vectorizamos 
descripcion = np.array(df['Descripcion raiz limpia']) # array para armar el bag of words
np.set_printoptions(precision=2)

# forma corta TFIDF vectorizer
vectorizador = TfidfVectorizer()
matriz_palabras = vectorizador.fit_transform(descripcion)
matriz_palabras = matriz_palabras.astype('float32') # cambiamos el tipo a float32 para disminuir uso de memoria
# %%
############ PCA ##############
df2 = pd.DataFrame(matriz_palabras.toarray())  # el array de matriz palabras pasamos a dataframe
df2.columns = vectorizador.get_feature_names() # agregamos nombres a las columnas con las palabras del vocabulario
# %%
pca = PCA(n_components=10000) # objeto de PCA con un máximo de 2000 componentes
pca = pca.fit(df2) # ajustamos el PCA al df2 de matriz de palabras
lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ] # generamos la lista de nombres de componentes del PCA
reduced_data = pca.transform(df2)  # aplicamos la transformación al dataframe de la matriz de palabras reduciendo la dimensionalidad
reduced_data = pd.DataFrame(reduced_data, columns = lista_PCA) # agregamos nombre de las columnas asociadas a los componentes del PCA
temp = df.reset_index() #df original reseteamos el indice para poder concatenar
df3 = pd.concat([temp,reduced_data], axis=1) #concatenamos dataframe original con componentes
df3.drop(['Descripcion raiz limpia'], axis=1, inplace=True) # eliminamos columna de descripcion

# %%
############ datos para train y test ###########
X = df3.drop(columns=['id_producto','label'], axis=1) # creamos la variables independientes
y = df3['label']  # creamos la variable dependiente

oversampling = SMOTE(sampling_strategy=0.30) # usamos oversampling sintético podemos elegir el nivel de oversampling con  sampling_strategy=0.80
X_balanceado, y_balanceado = oversampling.fit_resample(X, y) #Se obtienen nuevos X e y
# %%
'cambiamos a directorio de datos para guardar los dataframes creados'
os.chdir(r'Datos')
# %%
# guardamos el dataframe final para entrenamiento aplicando sobremuestreo SMOTE con sobremuestreo de 30%
train_test = {'X_balanceado':X_balanceado,'y_balanceado':y_balanceado}
fh = open('df_smote30_train_test.pkl','wb')
pickle.dump(reduced_data,fh)
fh.close()
# %%
'''
############# ARCHIVOS MUY PESADOS no guardar en GIT
# %%
#guardamos el datafra con la data original
fh = open('data.pkl','wb')
pickle.dump(data,fh)
fh.close()
# %%
#guardamos el dataframe aplicando de nltk para limpieza de campo descripcion
fh = open('df_nltk.pkl','wb')
pickle.dump(df,fh)
fh.close()
# %%
# guardamos el dataframe con la bolsa de palabras escalado con TF-IDF
fh = open('df_tfidf.pkl','wb')
pickle.dump(df2,fh)
fh.close()
# %%
# guardamos el dataframe con los componentes principales PCA
fh = open('df_PCA100.pkl','wb')
pickle.dump(df3,fh)
fh.close()
# %%
# guardamos el dataframe final previo a sobremuestreo SMOTE
fh = open('df_union.pkl','wb')
pickle.dump(df3,fh)
fh.close()
'''