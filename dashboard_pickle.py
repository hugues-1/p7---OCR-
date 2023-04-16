

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

import seaborn as sns

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


df=base

columns = list(df.columns)
dtypes = [str(dt) for dt in df.dtypes.tolist()]

# Créer un nouveau dataframe avec les noms de colonnes et les types
df_types = pd.DataFrame({
    'column_name': columns,
    'data_type': dtypes
})

# Sauvegarder les noms de colonnes et les types dans un fichier CSV
df_types.to_csv('types.csv', index=False)








feature_list =      ["CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","CNT_FAM_MEMBERS","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","APARTMENTS_AVG","DAYS_LAST_PHONE_CHANGE","AMT_REQ_CREDIT_BUREAU_YEAR","NAME_CONTRACT_TYPE_Cash loans","NAME_CONTRACT_TYPE_Revolving loans","CODE_GENDER_F","CODE_GENDER_M","FLAG_OWN_CAR_N","FLAG_OWN_CAR_Y","FLAG_OWN_REALTY_N","FLAG_OWN_REALTY_Y","NAME_INCOME_TYPE_Commercial associate","NAME_INCOME_TYPE_Pensioner","NAME_INCOME_TYPE_State servant","NAME_INCOME_TYPE_Unemployed","NAME_INCOME_TYPE_Working","NAME_INCOME_TYPE_nan","NAME_EDUCATION_TYPE_Academic degree","NAME_EDUCATION_TYPE_Higher education","NAME_EDUCATION_TYPE_Incomplete higher","NAME_EDUCATION_TYPE_Lower secondary","NAME_EDUCATION_TYPE_Secondary / secondary special","NAME_FAMILY_STATUS_Civil marriage","NAME_FAMILY_STATUS_Married","NAME_FAMILY_STATUS_Separated","NAME_FAMILY_STATUS_Single / not married","NAME_FAMILY_STATUS_Widow","NAME_HOUSING_TYPE_Co-op apartment","NAME_HOUSING_TYPE_House / apartment","NAME_HOUSING_TYPE_Municipal apartment","NAME_HOUSING_TYPE_Office apartment","NAME_HOUSING_TYPE_Rented apartment","NAME_HOUSING_TYPE_With parents","OCCUPATION_TYPE_Accountants","OCCUPATION_TYPE_Cleaning staff","OCCUPATION_TYPE_Cooking staff","OCCUPATION_TYPE_Core staff","OCCUPATION_TYPE_Drivers","OCCUPATION_TYPE_HR staff","OCCUPATION_TYPE_High skill tech staff","OCCUPATION_TYPE_IT staff","OCCUPATION_TYPE_Laborers","OCCUPATION_TYPE_Low-skill Laborers","OCCUPATION_TYPE_Managers","OCCUPATION_TYPE_Medicine staff","OCCUPATION_TYPE_Private service staff","OCCUPATION_TYPE_Realty agents","OCCUPATION_TYPE_Sales staff","OCCUPATION_TYPE_Secretaries","OCCUPATION_TYPE_Security staff","OCCUPATION_TYPE_Waiters/barmen staff","HOUSETYPE_MODE_block of flats","HOUSETYPE_MODE_specific housing","HOUSETYPE_MODE_terraced house","TARGET"]                   
# Liste déroulante pour sélectionner les features
graph = base[feature_list]


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
#X_resampled, y_resampled = RandomUnderSampler(random_state=22).fit_resample(X_train,y_train)

#model = RandomForestClassifier(max_depth=5, n_estimators=100,random_state=22).fit(X_resampled,y_resampled)
# del X_resampled,y_resampled,
del X,y,X_test,y_train,y_test


    
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
from streamlit_shap import st_shap

gc.collect()

# affichage simple des principaux critères
if st.sidebar.checkbox('affichage simple importance des critères', value=False):
    # Le code ci-dessous ne sera exécuté que si le bouton est coché
    st.write('Graphique explicatif ')
    explainer = shap.TreeExplainer(model,random_state=22)
    shap_values = explainer.shap_values(X_train)
    summary_plot = shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns, max_display=10, show=False)
    st_shap(summary_plot)
    gc.collect()
     
         
#affichage plus élaboré   
if st.sidebar.checkbox('affichage complet importance des critères', value=False):    
    st.write('Graphique explicatif ')
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_train.iloc[:,:])
    exp = Explanation(sv.values[:,:,1], 
                  sv.base_values[:,1], 
                  data=X_train.values, 
                  feature_names=X_train.columns)
    
    #shap.plots.beeswarm(shap_values, color=plt.get_cmap("cool"))
    # afficher le beeswarm plot
    #st_shap (shap.plots.beeswarm(exp,color=plt.get_cmap("cool")))
    summary_plot = shap.plots.beeswarm(exp)
    st_shap(summary_plot)
    gc.collect()
    
#demande du no client et stockage dans number

noclient= 0

if st.sidebar.checkbox('affichage de la probabilité d\'attribution pour un client', value=False):  

    st.write("Sélectionner le client ci_dessous, puis cliquez sur [Recherche] pour que l'API recherche l'enregistrement correspondant dans la base de donnée")
    x= st.slider("numéro client",0,999,1)
    

#converting the input in json
    inputs= {"nc":x}
    
  

    #on click fetch API
    if st.button('Recherche') :
       
        
        # Appel de l'API FastAPI pour récupérer les résultats
        response = requests.post("http://35.180.190.183:8080/noclient", data=json.dumps({"nc": x}))

    # Traitement de la réponse
        if response.ok:
            result = json.loads(response.content)
            st.write(f"Le résultat pour le client {x} est :")
            st.write(f"Résultat entier : {result['result_int']}")
            st.write(f"probabilité : {result['result_float']}")
            #if noclient!=0 : 
        #st.subheader( result_api)
            noclient= x
            st.subheader( noclient)
            explainer = shap.TreeExplainer(model, random_state=22)
            sv = explainer(X_train.iloc[noclient:noclient+1,:])
            
            exp = Explanation(sv.values[:,:,1], 
                  sv.base_values[:,1], 
                  data=X_train.values, 
                  feature_names=X_train.columns)
    
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig = waterfall(exp[0])
            st.pyplot(fig)
            plt.close(fig)
            
           
        else:
            st.write("Erreur lors de l'appel de l'API.")
              
    
    
    gc.collect()

if st.sidebar.checkbox('comparaison ', value=False) : 


    #selected_features = st.sidebar.selectbox('Selectionner une information', feature_list)
    selected_features = st.multiselect("Sélectionner deux données features", graph.columns)

    # Afficher le scatter plot avec la droite de corrélation
    if len(selected_features) == 2:
        fig, ax = plt.subplots()
        sns.regplot(data=graph, x=selected_features[0], y=selected_features[1], ax=ax)
    
   
        
        observation = graph.iloc[noclient]
        ax.scatter(observation[selected_features[0]], observation[selected_features[1]], marker="o", facecolors="none", edgecolors="r", s=200)

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Sélectionner deux indicateurs pour afficher le graphique.")

    gc.collect()

