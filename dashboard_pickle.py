

# Importer les bibliothèques nécessaires


# affichage des données générales des features importances globales 
#afficher les comparaisons proba versus deux variables ( plot type bundesliga ?) voir l'exemple ou l'énoncé 
# quand numéro client est obtenu feature importance affiché)  

"""import sys
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages/')
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages')
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages/streamlit/runtime/scriptrunner')
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/')
"""
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
#import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import imblearn as imb

from imblearn.under_sampling import RandomUnderSampler
#from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split

from shap import TreeExplainer

import gc
import json
import streamlit as st

import shap
import requests


base=pd.read_csv("~/mygit/p7---OCR-/base_sample.csv")
base = base.drop( columns = ['Unnamed: 0'])


# Adding an appropriate title for the test website
st.title("Tableau de bord client" )
st.title("Prêt à Dépenser")

st.write (" Choisissez une analyse a gauche ....")

# one_hot_encoder classique pour les non numériques
def one_hot_encoder(base, nan_as_category = True):
    original_columns = list(base.columns)
    categorical_columns = [col for col in base.columns if base[col].dtype == 'object']
    base2 = pd.get_dummies(base, columns= categorical_columns, dummy_na= True)
    new_columns = [c for c in base.columns if c not in original_columns]
    return base2
base2 =one_hot_encoder(base)
base = base2
del base2

base.to_csv('apresonehot.csv')

# Remplacer les valeurs manquantes par la moyenne de la colonne
base = base.fillna(base.mean())

# Séparer les variables explicatives (X) et la variable cible (y)
X = base.drop("TARGET", axis=1)
y = base["TARGET"]
del base

import pickle
with open('model.pickle', 'rb') as f:
    model, scaler = pickle.load(f)


# scaler 
col_names=X.select_dtypes(include='number').columns.tolist()
features = X[col_names]

features_scale = scaler.transform(features.values)
X[col_names] = features_scale
#print (X.shape) 
del features
del features_scale

#resample and fit the model ( a remplacer par un pickle) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_resampled, y_resampled = RandomUnderSampler(random_state=22).fit_resample(X_train,y_train)

#model = RandomForestClassifier(max_depth=5, n_estimators=100,random_state=22).fit(X_resampled,y_resampled)
del X_resampled,y_resampled,X,y,X_test,y_train,y_test


    
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
from streamlit_shap import st_shap

gc.collect()

# faire des pickle de shap_values et exp pour les récupérer ici 
if st.sidebar.checkbox('affichage simple importance des critères', value=False):
    # Le code ci-dessous ne sera exécuté que si le bouton est coché
    st.write('Graphique explicatif ')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    st_shap( shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns, max_display=10))

       
  
         
    
if st.sidebar.checkbox('affichage complet importance des critères', value=False):    
    st.write('Graphique explicatif ')
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_train.iloc[:,:])
    exp = Explanation(sv.values[:,:,1], 
                  sv.base_values[:,1], 
                  data=X_train.values, 
                  feature_names=X_train.columns)
    import matplotlib.pyplot as plt
    #shap.plots.beeswarm(shap_values, color=plt.get_cmap("cool"))
    # afficher le beeswarm plot
    st_shap (shap.plots.beeswarm(exp,color=plt.get_cmap("cool")))

    
#demande du no client et stockage dans number

noclient= 0

if st.sidebar.checkbox('affichage de la probabilité d\'attribution pour un client', value=False):  

    st.write("Sélectionner le client ci_dessous, puis cliquez sur [Recherche] pour que l'API recherche l'enregistrement correspondant dans la base de donnée")
    x= st.slider("numéro client",0,999,1)
    noclient = 0

#converting the input in json
    inputs= {"nc":x}

#st.write(json.dumps(inputs))

#on click fetch API
    if st.button('Recherche') :
        res = requests.post(url = "http://52.47.123.51:8080/noclient",data = json.dumps(inputs))
        #st.write ( res.text)
    #st.subheader(f"réponse API = {res.text}")
    #st.subheader(f"réponse API = {int(res.text)}")
        noclient=int(res.text)
        prob = model.predict_proba(X_train.iloc[noclient:noclient+1,:])*100
    #st.write (prob[0,1]) 
        st.subheader(f"réponse API probabilité de rejet du dossier = {prob[0,1]:.1f}%")
    

    
    if noclient!=0 : 

# TODO PLANTE A LA PREMIERE INTERROGATION qd noclient = 0 ? ?  
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_train.iloc[noclient:noclient+1,:])
        exp = Explanation(sv.values[:,:,1], 
                  sv.base_values[:,1], 
                  data=X_train.values, 
                  feature_names=X_train.columns)
    #idx = 0
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig = waterfall(exp[0])
        st.pyplot(fig)









