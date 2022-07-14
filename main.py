
import pandas as pd
import numpy as np
from funciones import *
#### librerias para NLP ##########
import re
import regex
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import  TfidfVectorizer, TfidfTransformer, CountVectorizer

###### PCA #######
from sklearn.decomposition import PCA

##########  libreria para entrenamiento #############



######### Carga, preprocesamiento datos ###################
'carga inicial dataset desde archivo csv a un objeto panda'
data = pd.read_csv(r'.\Datos\dataset_mercado_publico.csv', delimiter=';')
df = data.copy() # hacemos una copia sobre la cual trabajar
df = df.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion','Kupfer':'label'})
df.index = df['id_producto']  # cambiamos el indice del dataframe por el id_producto

'se eliminan columnas que no se utilizaran'
df.drop(columns=['id_licitacion','id_producto','Rubro2','Rubro3','nombre_producto'], inplace=True)

'limpieza de datos, nans, duplicados'
df.dropna(axis=0, inplace = True) #Si alguna fila tiene un NaN se elimina la fila 
df.drop_duplicates(keep='first', inplace=True) # elimina los registros duplicados

'creación de variables'
df = pd.get_dummies(df, columns=['Rubro1'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy


########### NLP #################
df['Descripcion limpia']= df['descripcion'].astype(str) #Convertimos la columna en string para poder trabajar con el texto


#Aplicamos la función texto_limpio para limpiar las descripciones
df['Descripcion limpia'] = df['Descripcion limpia'].apply(lambda texto: texto_limpio(texto)) 


# vamos a trabajar con las palabras estemizadas/raices
df['descripcion']= df['descripcion'].astype(str)
df['Descripcion raiz limpia']= df['descripcion'].apply(lambda texto: texto_raiz(texto)) #Aplicamos la función texto_raiz que nos convierte las palabras en sus raíces y las limpia
df.drop(columns=['descripcion','Descripcion limpia'], axis=1, inplace=True)


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
pca = PCA(n_components=2000) # objeto de PCA con un máximo de 2000 componentes
pca = pca.fit(df2) # ajustamos el PCA al df2 de matriz de palabras
lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ] # generamos la lista de nombres de componentes del PCA
reduced_data = pca.transform(df2)  # aplicamos la transformación al dataframe de la matriz de palabras reduciendo la dimensionalidad
reduced_data = pd.DataFrame(reduced_data, columns = lista_PCA) # agregamos nombre de las columnas asociadas a los componentes del PCA
temp = df.reset_index() #df original reseteamos el indice para poder concatenar
df3 = pd.concat([temp,reduced_data], axis=1) #concatenamos dataframe original con componentes
df3.drop(['Descripcion raiz limpia'], axis=1, inplace=True) # eliminamos columna de descripcion

############ datos para train y test ###########
