# %%
import pandas as pd

'carga inicial dataset desde archivo csv a un objeto panda'
data = pd.read_csv(r'.\Datos\dataset_mercado_publico.csv', delimiter=';')
df = data.copy() # hacemos una copia sobre la cual trabajar
df = df.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion','Kupfer':'label'})
df.index = df['id_producto']  # cambiamos el indice del dataframe por el id_producto
df0 = df.copy()

'se eliminan columnas que no se utilizaran'
df.drop(columns=['id_licitacion','id_producto','Rubro2','Rubro3','nombre_producto'], inplace=True)
df0.drop(columns=['id_licitacion','id_producto','descripcion'], inplace=True)

'limpieza de datos, nans, duplicados'
df.dropna(axis=0, inplace = True) #Si alguna fila tiene un NaN se elimina la fila 
df.drop_duplicates(keep='first', inplace=True) # elimina los registros duplicados

df0.dropna(axis=0, inplace = True) #Si alguna fila tiene un NaN se elimina la fila 
df0.drop_duplicates(keep='first', inplace=True) # elimina los registros duplicados


'creación de variables'
df = pd.get_dummies(df, columns=['Rubro1'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy

df0 = pd.get_dummies(df0, columns=['Rubro1'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['Rubro2'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['Rubro3'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy
df0 = pd.get_dummies(df0, columns=['nombre_producto'] ,drop_first=True) # convertimos var categorica Rubro1 en dummy