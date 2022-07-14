
#### librerias para NLP ##########
import re
import regex
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#### librerias para Métricas ##########
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, plot_roc_curve, plot_confusion_matrix, accuracy_score, roc_auc_score, make_scorer
import sklearn
import matplotlib.pyplot as plt
# %% 
sw = stopwords.words('spanish') # descargamos la lista de stopwords
sw.remove("no")
stemmer=SnowballStemmer("spanish")

# %% 
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


# %% 
# función para limpieza de texto (minusculas, quitar simbolos, quitar stopwords)
def texto_limpio(texto):
    texto = texto.lower() # convertir en minúsculas
    texto = re.sub(r"[\W\d_]+", " ",texto) # remover caract especiales y números
    texto = sin_tildes(texto) # remove tildes
    texto = texto.split() # tokenizar
    texto = [palabra for palabra in texto if len(palabra) > 3] # eliminar palabras con menos de 3 letras
    texto = [palabra for palabra in texto if palabra not in sw] # stopwords
    texto = " ".join(texto)
    return texto

# %% 

#Obtención de texto raíz limpio
def texto_raiz(texto):    
    texto = texto.lower() # convertir en minúsculas
    texto = re.sub(r"[\W\d_]+", " ",texto) # remover caract especiales y números
    texto = sin_tildes(texto) # remove tildes
    texto = texto.split() # tokenizar
    texto = [palabra for palabra in texto if len(palabra) > 3] # eliminar palabras con menos de 3 letras
    texto = [palabra for palabra in texto if palabra not in sw] # stopwords
    texto = [stemmer.stem(palabra) for palabra in texto] #stem
    texto = " ".join(texto)
    
    return texto


# %% 
#Obtención Métricas y threshold
def metrics(model, X_train, X_test, y_train, y_test, thr=0.5):
    
    plot_roc_curve(model, X_test, y_test) 
    plt.show()
# train
    probs=model.predict_proba(X_train)
    y_pred= (probs[:,1]>thr)*1
    print("Train AUC", sklearn.metrics.roc_auc_score(y_train, probs[:,1]))
    print(classification_report(y_train,y_pred,target_names=['0','1']))
    cm = confusion_matrix(y_train,y_pred) #Matriz
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

   
# test
    
    probs=model.predict_proba(X_test)
    y_pred= (probs[:,1]>thr)*1
    print("Test AUC", sklearn.metrics.roc_auc_score(y_test, probs[:,1]))
    print(classification_report(y_test,y_pred,target_names=['0','1']))
    cm = confusion_matrix(y_test,y_pred) #Matriz
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()