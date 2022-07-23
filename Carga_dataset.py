# %%
import os
import pickle
import pandas as pd

'carga inicial dataset desde archivo csv a un objeto panda'
data = pd.read_csv('.\Datos\dataset_mercado_publico.csv', delimiter=';')
data = data.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion','Kupfer':'label'})
data.index = data['id_producto']  # cambiamos el indice del dataframe por el id_producto
data.drop(columns=['id_producto'], inplace=True)

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

#guardamos el dataframe solo de las variables categoricas
X_cat = df0.drop(columns=['label'], axis=1) # creamos la variables independientes
y_cat = df0['label']  # creamos la variable dependiente
train_test_cat = {'X_cat':X_cat,'y_cat':y_cat}
print('data:',data.shape,'df0:',df0.shape,'df:',df.shape,'X_cat:',X_cat.shape,'y_cat:',y_cat.shape)