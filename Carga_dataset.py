import pandas as pd

data = pd.read_csv(r'.\Datos\dataset_mercado_publico.csv', delimiter=';')
df = data.copy()
df = df.rename(columns={'Tender_id':'id_licitacion','Item_Key':'id_producto','Nombre linea Adquisicion':'nombre_producto','Descripcion linea Adquisicion':'descripcion','Kupfer':'label'})
df.index = df['id_producto']
