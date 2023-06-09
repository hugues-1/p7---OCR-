#!/usr/bin/env python
# coding: utf-8

# ## ToDo : 

# In[ ]:


# Importer les bibliothèques nécessaires


import sys
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages/')
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages')
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages/streamlit/runtime/scriptrunner')
sys.path.append('home/ec2-user/miniconda3/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/')
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score
import numpy as np



from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import imblearn as imb
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from imblearn import FunctionSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from shap import TreeExplainer


# In[ ]:



import shap



base=pd.read_csv("base_sample.csv")
base = base.drop( columns = ['Unnamed: 0'])


# In[ ]:


# one_hot_encoder classique pour les non numériques
def one_hot_encoder(base, nan_as_category = True):
    original_columns = list(base.columns)
    categorical_columns = [col for col in base.columns if base[col].dtype == 'object']
    base2 = pd.get_dummies(base, columns= categorical_columns, dummy_na= True)
    new_columns = [c for c in base.columns if c not in original_columns]
    return base2
print (base.shape)
base2 =one_hot_encoder(base)
print (base2.shape)
base = base2
del base2


# In[ ]:


# Remplacer les valeurs manquantes par la moyenne de la colonne
base = base.fillna(base.mean())


# In[ ]:


# Séparer les variables explicatives (X) et la variable cible (y)
X = base.drop("TARGET", axis=1)
y = base["TARGET"]




# In[ ]:


# scaler 
col_names=X.select_dtypes(include='number').columns.tolist()
features = X[col_names]
scaler = StandardScaler().fit(features.values)
features_scale = scaler.transform(features.values)
X[col_names] = features_scale
print (X.shape) 
del features
del features_scale


# In[ ]:





# # fonction shap
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_resampled, y_resampled = RandomUnderSampler(random_state=22).fit_resample(X_train,y_train)
# entraînez votre modèle
#model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train, label=y_train), 100)
#model = DecisionTreeClassifier(random_state=22, max_depth = 3, min_samples_leaf = 2)
model = RandomForestClassifier(n_estimators=10, random_state=22)
model.fit(X_resampled,y_resampled)




# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()


# In[ ]:


from shap import TreeExplainer, Explanation
from shap.plots import waterfall


# In[ ]:


model = RandomForestClassifier(max_depth=5, n_estimators=100,random_state=22).fit(X_resampled,y_resampled)


# # explainer plotly waterfall 

# In[ ]:


import plotly as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import streamlit as st


# In[ ]:


data = {'feature': [], 'row_id': [], 'probability': [], 'shap_value': [], 'feature_value': []}

idx = 4


explainer = TreeExplainer(model)




sv = explainer(X_train.iloc[[idx], :], check_additivity=True)
prob = model.predict_proba(X_train.iloc[[idx], :])[0][1]
for col in range(X_train.shape[1]):
    data['feature'].append(X_train.columns[col])
    data['row_id'].append(idx)
    data['probability'].append(prob)
    data['shap_value'].append(sv.values[0, col, 1])
    data['feature_value'].append(X_train.iloc[idx, col])
        
df = pd.DataFrame(data)
df_pivot = df.pivot(index='feature', columns='row_id', values='shap_value')
df_pivot.columns.name = None
df_pivot.reset_index(inplace=True)
df_pivot['sum'] = df_pivot.iloc[:, 1:].sum(axis=1)

waterfall_data = df_pivot.sort_values('sum', ascending=False).head(20)[['feature', 'sum']].copy()
waterfall_data.columns = ['Feature', 'Contribution']
waterfall_data['Contribution'] = waterfall_data['Contribution'] / 100

waterfall_data_neg = df_pivot.sort_values('sum', ascending=True).head(20)[['feature', 'sum']].copy()
waterfall_data_neg.columns = ['Feature', 'Contribution']
waterfall_data_neg['Contribution'] = waterfall_data_neg['Contribution'] / 100 * 1



waterfall_data = pd.concat([waterfall_data_neg, waterfall_data], ignore_index=True)

""" fig = go.Figure(go.waterfall(waterfall_data, x='Feature', y='Contribution', color=waterfall_data['Contribution']>0, orientation='v',
                   text='Contribution', hover_data={'Contribution': ':.2f'}))""" 
fig = go.Figure(go.Waterfall(
    name = "Feature Contributions",
    orientation = "h",
    measure = list(waterfall_data['Contribution']),
    y = list(waterfall_data['Feature']),
    textposition = "outside",
    x = list(waterfall_data['Contribution']),
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    decreasing = {"marker":{"color":"green"}},
    increasing = {"marker":{"color":"red"}},
    totals = {"marker":{"color":"deep sky blue", "line":{"color":"blue", "width":3}}},
    base = 0
))

fig.update_layout(
    
    xaxis_title = "contribution",
    yaxis_title = "indicateur",
    height = 500,
    font = dict(
        family = "Arial, sans-serif",
        size = 14,
        color = "#000000"
    )
)



#fig.update_layout(title_text='Feature Contributions for Sample 42', title_x=0.5, 'score: ',prob)
fig.update_layout(title_text='Contributions pour le client  {} (Prob = {:.2f}%)'.format(idx, prob*100), title_x=0.5)
# fig.show()
st.plotly_chart(fig)



# # Explainer Shap Waterfall

# In[ ]:


X_sample = X_train.sample(1000)


# In[ ]:


X_sample.shape


# In[ ]:


explainer = TreeExplainer(model)
sv = explainer(X_train.iloc[4:5,:])
exp = Explanation(sv.values[:,:,1], 
                  sv.base_values[:,1], 
                  data=X_train.values, 
                  feature_names=X_train.columns)
idx = 0
fig = waterfall(exp[idx])
st.pyplot(fig)

# In[ ]:


#shap_values = explainer.shap_values(X_sample)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#fig = shap.force_plot(explainer.expected_value[0], shap_values[0])
#st.pyplot(fig)
#st_shap (shap.force_plot(explainer.expected_value[0], shap_values[0]))
#shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0,:])
#st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))


# In[ ]:



