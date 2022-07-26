# %%
####### funciones, carga de datos inicial
from Carga_dataset import data,df,df0,train_test_cat
from Funciones.funciones import texto_limpio,texto_raiz
from collections import Counter
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

'cambiamos a directorio de datos para guardar los dataframes creados'
os.chdir(r'Datos')
# %%

#guardamos el dataframe con la data original
fh1 = open('data.pkl','wb')
pickle.dump(data,fh1)
fh1.close()

fh0 = open('.\df_categorias.pkl','wb')
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
#df.drop(columns=['descripcion','Descripcion limpia'], axis=1, inplace=True)

# %%
#guardamos el dataframe aplicando de nltk para limpieza de campo descripcion
fh2 = open('df_nltk.pkl','wb')
pickle.dump(df,fh2)
fh2.close()
# %%
'bag of words (vector de palabras) y escalamos con TF-IDF'
# ahora vectorizamos 
vocabulario = np.array(df['Descripcion raiz limpia']) # array para armar el bag of words
np.set_printoptions(precision=2)

fh3 = open('vocabulario.pkl','wb')
pickle.dump(vocabulario,fh3)
fh3.close()

# forma corta TFIDF vectorizer o bolsa de palabras normalizada
vectorizador = TfidfVectorizer()
matriz_palabras = vectorizador.fit_transform(vocabulario) # creamos la bolsa de palabras
matriz_palabras = matriz_palabras.astype('float32') # cambiamos el tipo a float32 para disminuir uso de memoria

fh4 = open('TFIDF.pkl','wb')
pickle.dump(vectorizador,fh4)
fh4.close()
# %%
############ PCA ##############
df2 = pd.DataFrame(matriz_palabras.toarray())  # el array de matriz palabras pasamos a dataframe
df2.columns = vectorizador.get_feature_names() # agregamos nombres a las columnas con las palabras del vocabulario
#54484 registros x 19535 columnas ojo que el indice es numérico con el mismo orden que el dataframe original
# %%
############# ARCHIVOS MUY PESADOS para guardar en GIT o que no se usan 
pca = PCA(n_components=10000) # objeto de PCA con un máximo de 10000 componentes
#pca = PCA(n_components=1000) 
#pca = PCA(n_components=2000) 
#pca = PCA(n_components=5000) 
#pca = PCA(n_components=7000) 
pca = pca.fit(df2) # ajustamos el PCA al df2 de matriz de palabras
lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ] # generamos la lista de nombres de componentes del PCA
dfPCA = pca.transform(df2)  # aplicamos la transformación al dataframe de la matriz de palabras reduciendo la dimensionalidad
dfPCA = pd.DataFrame(dfPCA, columns = lista_PCA) # agregamos nombre de las columnas asociadas a los componentes del PCA

fh4 = open('PCA.pkl','wb')
pickle.dump(pca,fh4)
fh4.close()

# %%
# guardamos el dataframe con los componentes principales PCA
fh5 = open('df_PCA10k.pkl','wb')
#fh4 = open('df_PCA1k.pkl','wb')
#fh4 = open('df_PCA2k.pkl','wb')
#fh4 = open('df_PCA5k.pkl','wb')
#fh4 = open('df_PCA7k.pkl','wb')
pickle.dump(dfPCA,fh5)
fh5.close()

# %%
########## Unimos los PCA con la variable categorica
# 1ro filtramos el dataframe de variables categoricas luego de haber aplicado catboost
fh8=open('array_categorias_importantes.pkl','rb')
filtro=pickle.load(fh8)
fh8.close()

dfcat = df0[filtro]  #dataframe filtrado solo por las columnas
dfcat = dfcat.reset_index() #df0 dataframe con variables categoricas reseteamos el indice para poder concatenar
dfcat.head()

fh9=open('df_PCA10k.pkl','rb')  #aseguramos de cargar los PCA10mil
dfPCA=pickle.load(fh9)
fh9.close()
# %%
dfcat.shape
# %%

# creamos la variables independientes
X = pd.concat([dfcat,dfPCA], axis=1) # concatenamos las matrices de variables categoricas seleccionadas con el PCA de palabras
X.index = X['id_producto']  #vuelvo a dejar el índice del producto
X.drop(columns=['id_producto'], inplace=True) # elimino la columna 
# creamos la variable dependiente
y = data[['label']]  

print('data:',data.shape,'df0:',df0.shape,'dfPCA:',dfPCA.shape)
print('X:',X.shape,'y:',y.shape)
# %%
# guardamos el dataframe final para entrenamiento aplicando sobremuestreo SMOTE con sobremuestreo de 30%
train_test = {'X':X,'y':y}
fh5 = open('df_PCA10k_train_test.pkl','wb')  ####### OJO QUE ARCHIVO PESA 2GB
#fh = open('df_PCA10_train_test.pkl','wb')
pickle.dump(train_test,fh5)
fh5.close()
# %%
##################### SOLO PARA EXPORTAR A COLAB
##### guardamos las variables como numpy arrays 
X = train_test['X'].values
y = train_test['y'].values
print('X:',X.shape,'y:',y.shape)
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