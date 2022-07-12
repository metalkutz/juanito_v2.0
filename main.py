import pandas as pd
import numpy as np

#### librerias para NLP ##########
import re
import regex
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

##########  libreria para entrenamiento #############



######### Carga, preprocesamiento datos ###################
'carga inicial dataset desde archivo csv a un objeto panda'
data = pd.read_csv(r'.\Datos\dataset_mercado_publico.csv', delimiter=';')
df = data.copy() # hacemos una copia sobre la cual trabajar
df = df.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion','Kupfer':'label'})
df.index = df['id_producto']  # cambiamos el indice del dataframe por el id_producto

'se eliminan columnas que no se utilizaran'
df.drop(columns=['id_licitacion','id_producto','Rubro2','Rubro3'], inplace=True)

'limpieza de datos, nans, duplicados'
df.dropna(axis=0, inplace = True) #Si alguna fila tiene un NaN se elimina la fila 
df.drop_duplicates(keep='first', inplace=True) # elimina los registros duplicados

'creación de variables'
df = pd.get_dummies(df, columns=['Rubro1'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy

########### NLP #################
df['Descripcion limpia']= df['descripcion'].astype(str) #Convertimos la columna en string para poder trabajar con el texto

sw = stopwords.words('spanish') # descargamos la lista de stopwords
sw.remove("no")

# función para limpiar tildes
def sin_tildes(s):
    tildes = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for origen, destino in tildes:
        s = s.replace(origen, destino)
    return s

# función para limpieza de texto (minusculas, quitar simbolos, quitar stopwords)
def texto_limpio(texto):
    texto = texto.lower() # convertir en minúsculas
    texto = re.sub(r"[\W]+", " ",texto) # remover caract especiales
    texto = sin_tildes(texto) # remove tildes
    texto = texto.split() # tokenizar
    texto = [palabra for palabra in texto if palabra not in sw] # stopwords
    texto = " ".join(texto)
    return texto

df['Descripcion limpia'] = df['Descripcion limpia'].apply(lambda texto: texto_limpio(texto)) #Aplicamos la función texto_limpio para limpiar las descripciones

stemmer=SnowballStemmer("spanish") #Obtención de texto raíz limpio

#función que adiciona el convertir la palabra y la estemiza
def texto_raiz(texto):    
    texto = texto.lower() # convertir en minúsculas
    texto = re.sub(r"[\W]+", " ",texto) # remover caract especiales
    texto = sin_tildes(texto) # remove tildes
    texto = texto.split() # tokenizar
    texto = [palabra for palabra in texto if palabra not in sw] # stopwords
    texto = [stemmer.stem(palabra) for palabra in texto] #stem
    texto = " ".join(texto)
    return texto

df['Descripcion raiz limpia']= df['Descripcion limpia'].apply(lambda texto: texto_raiz(texto)) #Aplicamos la función texto_raiz que nos convierte las palabras en sus raíces

