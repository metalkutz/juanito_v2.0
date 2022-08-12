import pickle

##### archivo generado con el codigo dataset_train_test.py
fh1=open('.\Datos\data.pkl','rb')
data=pickle.load(fh1)
fh1.close()
data.head()

fh0=open('.\Datos\df_categorias.pkl','rb')
train_test_cat=pickle.load(fh0)
fh0.close()

X_cat = train_test_cat['X_cat']
y_cat = train_test_cat['y_cat']
print('X_cat:',X_cat.shape,'y_cat:',y_cat.shape)

fh2=open('.\Datos\df_nltk.pkl','rb')
df=pickle.load(fh2)
fh2.close()
df.head()

fh4=open('.\Datos\df_PCA10k.pkl','rb')
reduced_data=pickle.load(fh4)
fh4.close()
reduced_data.head()

fh5=open('.\Datos\df_PCA10k_train_test.pkl','rb')
train_test=pickle.load(fh5)
fh5.close()
X = train_test['X']
y = train_test['y']

#### Funciones para preprocesamiento de la data
fh3 = open('.\Funciones\Filtro_cat.pkl','rb')
filtro = pickle.load(fh3)
fh3.close()

fh6 = open('.\Funciones\TFIDF.pkl','rb')
vectorizador = pickle.load(fh6)
fh6.close()

fh7 = open('.\Funciones\PCA.pkl','rb')
pca = pickle.load(fh7)
fh7.close()

#### Modelos para predicción
fh8 = open('.\Modelos\m_reglog_PCA_final.pkl','rb')
logreg_model = pickle.load(fh8)
fh8.close()

fh9 = open('.\Modelos\m_XGBoost_PCA_vfinal.pkl','rb')
xgboost_model = pickle.load(fh9)
fh9.close()