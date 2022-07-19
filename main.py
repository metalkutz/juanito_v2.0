# %%
####### funciones, carga de datos inicial
from funciones import* 

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
'''fh3 = open('df_tfidf.pkl','wb')
pickle.dump(df2,fh3)
fh3.close()
df2.head()'''
############ PCA ##############
##### archivo generado con el codigo dataset_train_test.py
'''fh4=open('df_PCA100.pkl.pkl','rb')
dff3=pickle.load(fh4)
fh4.close()
df3.head()'''
############ datos para train y test ###########
##### archivo generado con el codigo dataset_train_test.py
fh5=open('df_smote30_train_test.pkl','rb')
train_test=pickle.load(fh5)
fh5.close()


X_balanceado = train_test['X_balanceado']
y_balanceado = train_test['y_balanceado']
print('X:',X_balanceado.shape,'y:',y_balanceado.shape)

# %%
##### archivo reducido en componentes PCA = 100 para prueba de flujos
'''fh6=open('df_smote30_train_test(redux).pkl','rb')
train_test_redux=pickle.load(fh6)
fh6.close()


X_balanceado_redux = train_test_redux['X_balanceado']
y_balanceado_redux = train_test_redux['y_balanceado']
print('X:',X_balanceado_redux.shape,'y:',y_balanceado_redux.shape)'''
# %%
