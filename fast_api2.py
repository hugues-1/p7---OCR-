
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


base=pd.read_csv("/home/ec2-user/mygit/p7---OCR-/base_sample.csv")
#base=pd.read_csv("~/mygit/p7---OCR-/base_sample.csv")
base = base.drop( columns = ['Unnamed: 0'])

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
with open('/home/ec2-user/model.pickle', 'rb') as f:
    model, scaler = pickle.load(f)


# scaler 
col_names=X.select_dtypes(include='number').columns.tolist()
features = X[col_names]

features_scale = scaler.transform(features.values)
X[col_names] = features_scale
#print (X.shape) 
del features
del features_scale

#resample 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=22)
X_resampled, y_resampled = RandomUnderSampler(random_state=22).fit_resample(X_train,y_train)
del X_resampled,y_resampled,X,y,X_test,y_train,y_test

class User_input(BaseModel):
    nc : int
    #nc : float   
    
app = FastAPI()

@app.post("/noclient")
def operate(input:User_input):
      
    
    resultpredict = model.predict(X_train.iloc[input.nc:input.nc+1,:])
    result_int = int(resultpredict[0])
    print ( result_int)
    resultproba = model.predict_proba(X_train.iloc[input.nc:input.nc+1,:])
    result_float = float(resultproba[0,1])
    print (result_float)
                    
    
    return {"result_int": result_int, "result_float": result_float}
