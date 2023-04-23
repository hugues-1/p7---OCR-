# p7 - OCR - Projet Prêt à Dépenser

## Objectif : 
- Développer un dashboard interactif permettant aux chargés de relation client d’expliquer les décisions
- Permettre aux clients d’explorer facilement leurs informations personnelles.

## Mission applicatif : 
- Construire un modèle de scoring
- Construire un dashboard interactif à destination des gestionnaires 
- Mettre en production le modèle de scoring et le dashboard avec des API

## Mise en oeuvre  : 
- Utiliser un kernel externe pour l’analyse exploratoire et le feature engineering : **lightGBM2.ipynb**
- Déterminer le meilleur modeles en testant plusieurs algorithmes, leurs hyperparametres, les méthodes de correction du déséquilibre de la base afin de mettre en oeuvre un modele de sélection des clients tenant compte des parametres métier: 
    - **prediction_calc_nan.ipynb**
    - **result_score_metier_clean.ipynb** 
    - **MLflow-surf.ipynb**
- Mettre en place le dashboard : **dashboard_pickle.py**
- Mettre en place l'API : **fast_api2.py**
- Anticiper sur le DataDrift  **evidently.ipynb**

## Outils utilisés
- environnement DataScience JupyterLab Scikit-learn, Pandas ..
- MLflow recording et visualisation d'expérimentations
- Streamlit pour le dashboard interactif 
- FastApi pour le scoring client et le transmettre au dashboard
- Screen pour assurer l'exécution des crtips dashboard et et api en arrière plan sur EC2
- Evidently pour mener l'anayse DataDrift et DataQuality
- GitHub pour la gestion des scripts et des modifications

## Plate-forme
- en local pour les modéles de prédiction et les analyses 
- sur AWS EC2 pour héberger le dashboard et l'api

 
