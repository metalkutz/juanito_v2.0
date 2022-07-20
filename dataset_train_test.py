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
from sklearn.model_selection import train_test_split
###### PCA #######
from sklearn.decomposition import PCA

######### re-balanceo de muestras ##########
from imblearn.over_sampling import SMOTE 
# %%
'cambiamos a directorio de datos para guardar los dataframes creados'
os.chdir(r'Datos')

# %%
####### desde archivo Carga_dataset se trae dataframe con
'carga inicial dataset desde archivo csv a un objeto panda'
'se eliminan columnas que no se utilizaran'
'limpieza de datos, nans, duplicados'
'creación de variables'

#guardamos el dataframe con la data original
fh1 = open('data.pkl','wb')
pickle.dump(data,fh1)
fh1.close()
# %%
#guardamos el dataframe solo de las variables categoricas
X_cat = df0.drop(columns=['label'], axis=1) # creamos la variables independientes
y_cat = df0['label']  # creamos la variable dependiente
train_test_cat = {'X_cat':X_cat,'y_cat':y_cat}

fh0 = open('df_categorias.pkl','wb')
pickle.dump(train_test_cat,fh0)
fh0.close()
# %%
########### NLP #################
df['Descripcion limpia']= df['descripcion'].astype(str) #Convertimos la columna en string para poder trabajar con el texto

#Aplicamos la función texto_limpio para limpiar las descripciones
df['Descripcion limpia'] = df['Descripcion limpia'].apply(lambda texto: texto_limpio(texto)) 

# vamos a trabajar con las palabras estemizadas/raices
df['descripcion']= df['descripcion'].astype(str)
df['Descripcion raiz limpia']= df['descripcion'].apply(lambda texto: texto_raiz(texto)) #Aplicamos la función texto_raiz que nos convierte las palabras en sus raíces y las limpia
df.drop(columns=['descripcion','Descripcion limpia'], axis=1, inplace=True)

#guardamos el dataframe aplicando de nltk para limpieza de campo descripcion
fh2 = open('df_nltk.pkl','wb')
pickle.dump(df,fh2)
fh2.close()
# %%
fh2=open('df_nltk.pkl','rb')
df=pickle.load(fh2)
fh2.close()
df.head()
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
############# ARCHIVOS MUY PESADOS para guardar en GIT o que no se usan 
'''# guardamos el dataframe con la bolsa de palabras escalado con TF-IDF
fh3 = open('df_tfidf.pkl','wb')
pickle.dump(df2,fh3)
fh3.close()'''
# %%
pca = PCA(n_components=10000) # objeto de PCA con un máximo de 2000 componentes
pca = pca.fit(df2) # ajustamos el PCA al df2 de matriz de palabras
lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ] # generamos la lista de nombres de componentes del PCA
reduced_data = pca.transform(df2)  # aplicamos la transformación al dataframe de la matriz de palabras reduciendo la dimensionalidad
reduced_data = pd.DataFrame(reduced_data, columns = lista_PCA) # agregamos nombre de las columnas asociadas a los componentes del PCA

# guardamos el dataframe con los componentes principales PCA
fh4 = open('df_PCA10k.pkl','wb')
pickle.dump(reduced_data,fh4)
fh4.close()

# %%
# Unimos los PCA con la variable categorica
temp = df.reset_index() #df original reseteamos el indice para poder concatenar
df3 = pd.concat([temp,reduced_data], axis=1) #concatenamos dataframe original con componentes
df3.drop(['Descripcion raiz limpia'], axis=1, inplace=True) # eliminamos columna de descripcion

# guardamos el dataframe final previo a sobremuestreo SMOTE
'''fh = open('df_union.pkl','wb')
pickle.dump(df3,fh)
fh.close()'''

############ datos para train y test ###########
X = df3.drop(columns=['id_producto','label'], axis=1) # creamos la variables independientes
y = df3['label']  # creamos la variable dependiente

# guardamos el dataframe final para entrenamiento aplicando sobremuestreo SMOTE con sobremuestreo de 30%
train_test = {'X':X,'y':y}
fh5 = open('df_PCA10k_train_test.pkl','wb')  ####### OJO QUE ARCHIVO PESA 2GB
#fh = open('df_PCA10_train_test.pkl','wb')
pickle.dump(train_test,fh5)
fh5.close()
# %%
fh5=open('df_PCA10k_train_test.pkl','rb')
train_test=pickle.load(fh5)
fh5.close()

X = train_test['X'].values
y = train_test['y'].values
print('X:',X.shape,'y:',y.shape)
# %%
###### Generamos pickle para exportar tanto X_train y el y_train  a Colab y poder ejecutar distintos modelos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

oversampling = SMOTE(sampling_strategy=0.30) # usamos oversampling sintético podemos elegir el nivel de oversampling con  sampling_strategy=0.80
X_train_smote, y_train_smote = oversampling.fit_resample(X_train, y_train) #Se obtienen nuevos X e y

#### guardamos el narray en un pickle 
train = {'X_train_smote':X_train_smote,'y_train_smote':y_train_smote}
#fh7 = open('df_PCA10k_smote_train.pkl','wb') 
fh7 = open('df_PCA10k_smote_train_np.pkl','wb') 
pickle.dump(train,fh7)
fh7.close()

print('X:',len(X_train_smote),'x',len(X_train_smote[0]),'y:',len(y_train_smote),'x',)