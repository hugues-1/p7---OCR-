

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
import matplotlib.image as mpimg

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

#avant one hot features lisibles 

cols_numeriques = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_REGISTRATION','OWN_CAR_AGE','CNT_FAM_MEMBERS','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','TOTALAREA_MODE','DAYS_LAST_PHONE_CHANGE','CNT_CHILDREN','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_ID_PUBLISH']

cols_non_numeriques = ['HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','TARGET']

basegraph= base

# Charger l'image
logo = mpimg.imread("logo_pad.png")

# Afficher l'image
#st.image(logo)
st.image(logo, use_column_width=False, width=150)



# Adding an appropriate title for the test website
st.title("Tableau de bord client" )
st.title("Prêt à Dépenser")




st.write (" Choisissez une analyse à gauche ....")

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

#apresonehot selection des features compréhensible 
feature_list =      ["CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","CNT_FAM_MEMBERS","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","APARTMENTS_AVG","DAYS_LAST_PHONE_CHANGE","AMT_REQ_CREDIT_BUREAU_YEAR","NAME_CONTRACT_TYPE_Cash loans","NAME_CONTRACT_TYPE_Revolving loans","CODE_GENDER_F","CODE_GENDER_M","FLAG_OWN_CAR_N","FLAG_OWN_CAR_Y","FLAG_OWN_REALTY_N","FLAG_OWN_REALTY_Y","NAME_INCOME_TYPE_Commercial associate","NAME_INCOME_TYPE_Pensioner","NAME_INCOME_TYPE_State servant","NAME_INCOME_TYPE_Unemployed","NAME_INCOME_TYPE_Working","NAME_INCOME_TYPE_nan","NAME_EDUCATION_TYPE_Academic degree","NAME_EDUCATION_TYPE_Higher education","NAME_EDUCATION_TYPE_Incomplete higher","NAME_EDUCATION_TYPE_Lower secondary","NAME_EDUCATION_TYPE_Secondary / secondary special","NAME_FAMILY_STATUS_Civil marriage","NAME_FAMILY_STATUS_Married","NAME_FAMILY_STATUS_Separated","NAME_FAMILY_STATUS_Single / not married","NAME_FAMILY_STATUS_Widow","NAME_HOUSING_TYPE_Co-op apartment","NAME_HOUSING_TYPE_House / apartment","NAME_HOUSING_TYPE_Municipal apartment","NAME_HOUSING_TYPE_Office apartment","NAME_HOUSING_TYPE_Rented apartment","NAME_HOUSING_TYPE_With parents","OCCUPATION_TYPE_Accountants","OCCUPATION_TYPE_Cleaning staff","OCCUPATION_TYPE_Cooking staff","OCCUPATION_TYPE_Core staff","OCCUPATION_TYPE_Drivers","OCCUPATION_TYPE_HR staff","OCCUPATION_TYPE_High skill tech staff","OCCUPATION_TYPE_IT staff","OCCUPATION_TYPE_Laborers","OCCUPATION_TYPE_Low-skill Laborers","OCCUPATION_TYPE_Managers","OCCUPATION_TYPE_Medicine staff","OCCUPATION_TYPE_Private service staff","OCCUPATION_TYPE_Realty agents","OCCUPATION_TYPE_Sales staff","OCCUPATION_TYPE_Secretaries","OCCUPATION_TYPE_Security staff","OCCUPATION_TYPE_Waiters/barmen staff","HOUSETYPE_MODE_block of flats","HOUSETYPE_MODE_specific housing","HOUSETYPE_MODE_terraced house","TARGET"]                   
# Liste déroulante pour sélectionner les features
graph = base[feature_list]


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

if st.sidebar.checkbox('sélection d\'un client puis affichage de la probabilité d\'attribution pour un client', value=False):  

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
            #st.write(f"Résultat entier : {result['result_int']}")
            st.write(f"Score client : {round(result['result_float']*100,0)} %")
            st.write("si le score de risque client est inférieur à 50%, le dossier est accepté")
        #st.subheader( result_api)
            noclient= x
            st.subheader("critères les plus déterminants pour le client : "+str( noclient))
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

if st.sidebar.checkbox('affichage des données et positionnement du client', value=False) : 

    #selected_features = st.sidebar.selectbox('Selectionner une information', feature_list)
    selected_features = st.multiselect("Sélectionner deux données features", graph.columns)
    st.write ( "par exemple : AMT_INCOME et AMT_CREDIT, le cercle rouge indique le positionnement du client sélectionné")
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

  
  
    
def afficher_tableau_ecart_colonne(basegraph: pd.DataFrame, noclient: int):
   
  
    st.subheader("Numéro client : " + str(noclient))
    st.write ( basegraph.shape)
  
    # Extraction des données numériques du client sélectionné
    donnees_client = basegraph.loc[noclient, cols_numeriques]
    
    
    # Calcul de la moyenne des autres clients
    moyenne_autres_clients = basegraph.loc[basegraph.index != noclient, cols_numeriques].mean()
    
    # Calcul du pourcentage d'écart
    pourcent = donnees_client / moyenne_autres_clients * 100
    
      

    # Concaténation des données dans un dataframe
    comp = pd.concat([donnees_client, moyenne_autres_clients, pourcent], axis=1)
    comp.columns = ['Données client', 'Moyenne autres clients', 'Ecart en % par rapport à la moyenne']
    comp.index.name = 'Caractéristiques'

    # Formatage du tableau pour afficher 2 chiffres après la virgule
    comp_mef = comp.style.format("{:,.2f}")

    # Affichage du tableau
    st.write(comp_mef)
    st.subheader( "Ecart en % par rapport à la moyenne" ) 
    # Affichage des données numériques et des moyennes dans un graphique
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    #pourcent.plot.barh(ax=ax, color='g', alpha=0.5, label='Ecart en % par rapport à la moyenne')
    pourcent.plot.barh(ax=ax, color='g', alpha=0.5)
    ax.legend()
    ax.xaxis.set_label_position('top')
    st.pyplot(fig)

if st.sidebar.checkbox('comparaison par rapport aux autre clients  ', value=False) :  
    
    afficher_tableau_ecart_colonne(basegraph,noclient) 
    
if st.sidebar.checkbox('comparaison par rapport à une autre typologie de client ', value=False) :  
      
      # Sélection de la colonne non numérique
    col_non_numerique = st.selectbox("Sélectionnez une colonne non numérique :", cols_non_numeriques)
    
    if col_non_numerique:
        valeurs_col_non_numerique = st.multiselect("Sélectionnez une ou plusieurs valeurs :", basegraph[col_non_numerique].unique())
        
    
    # Sélection des valeurs dans la colonne non numérique
        if valeurs_col_non_numerique :
    
            # Filtrage des lignes correspondant aux valeurs sélectionnées
            base_filtree = basegraph[basegraph[col_non_numerique].isin(valeurs_col_non_numerique)]
            # Ajout de la ligne correspondant au client sélectionné
            if noclient not in base_filtree.index:
                base_filtree = pd.concat([base_filtree, basegraph.loc[[noclient]]])
            
            afficher_tableau_ecart_colonne(base_filtree,noclient)
                                    
