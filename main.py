# %%
####### os, pickle
import os
import pickle

###### pandas, numpy, funciones
import pandas as pd
import numpy as np

##########  libreria para entrenamiento #############


# %%
######### Carga, preprocesamiento datos ###################
os.chdir(r'Datos')
# %%
##### archivo generado con el codigo dataset_train_test.py
fh1=open('data.pkl','rb')
data=pickle.load(fh1)
fh1.close()
data.head()
# %%
##### archivo generado con el codigo dataset_train_test.py
fh0=open('df_categorias.pkl','rb')
train_test_cat=pickle.load(fh0)
fh0.close()

X_cat = train_test_cat['X_cat']
y_cat = train_test_cat['y_cat']
print('X_cat:',X_cat.shape,'y_cat:',y_cat.shape)
# %%
########### NLP ################# 
##### archivo generado con el codigo dataset_train_test.py
fh2=open('df_nltk.pkl','rb')
df=pickle.load(fh2)
fh2.close()
df.head()
# %%
# ahora vectorizamos y creamos el bag of words

# forma corta TFIDF vectorizer
##### archivo generado con el codigo dataset_train_test.py
'''fh3 = open('df_tfidf.pkl','rb')
df_tfidf=pickle.load(df_tfidf,fh3)
fh3.close()
df_tfidf.head()'''
############ PCA ##############
##### archivo generado con el codigo dataset_train_test.py
fh4=open('df_PCA10k.pkl','rb')
reduced_data=pickle.load(fh4)
fh4.close()
reduced_data.head()

# %%
############ datos para train y test ###########
##### archivo generado con el codigo dataset_train_test.py
fh5=open('df_PCA10k_train_test.pkl','rb')
train_test=pickle.load(fh5)
fh5.close()

X = train_test['X']
y = train_test['y']
print('X:',X.shape,'y:',y.shape)
# %%
'''
##### archivo reducido en componentes PCA = 10 para prueba de flujos
fh6=open('df_PCA10_train_test.pkl','rb')
train_test_redux=pickle.load(fh6)
fh6.close()

X_redux = train_test_redux['X']
y_redux = train_test_redux['y']
print('X_redux:',X_redux.shape,'y_redux:',y_redux.shape)'''
# %%
