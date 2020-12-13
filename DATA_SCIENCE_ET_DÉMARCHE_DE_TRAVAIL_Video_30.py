# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:41:07 2020

@author: Administrator
"""
print("DATA SCIENCE ET DÉMARCHE DE TRAVAIL")

#==============DATA SCIENCE ET DÉMARCHE DE TRAVAIL ========

"""
voir image_01,02,03

#==========Définir un objectif mesurable:
-Objectif: prédire si une personne est infectée en fonction des données cliniques disponibles
-Métrique : F1 -> 50% et Recall -> 70%

pour commencer il nous faut toujours se fixé un objectif mesurable:
    Par exemple dans notre projet de covid_19, on pouvait avoir comme objectif de 
    predire si une personne est infectée ou non avec 90% d'exactitude.
    autrement dit sur 100 prédictions, on a raison 90% du temps. Mtn le problème
    dans ces genres de projet  on a des classes déséquilibrée c'est à dire sur 100
    individus 90% d'entre eux ne vont pas avoir de virus c'est à dire vont etre dans 
    la classe zéro et 10% d'entre eux vont avoir le virus. Donc vont etre dans la 
    classe A.
    Alors à quoi c'est un problème?
    C'est que il est plutot facile d'atteindre 90% de performances. Il suffit de
    développer un modèle qui disent que personne dans la vie n'est infecté par le coronavirus
    du coup sur 100 prédictions le modèle aura raison 90% du temps.
    Est ce que c'est un modèle pourtant?
    Bien sur que non
    Du coup dans ce genre de circonstances, il nous faut choisir une autre mesure de
    performance que l'exactitude en l'occurrence de bonne mesures de performances vont
    etre la précision, le recall aussi appelé sensibilité ou le score F1 qui fait le rapport entre
    entre la précision et le recall
    
Une fois qu'on s'est fixé un objectif, on va pouvoir passer au travail et en général notre travail de
data scientist est divisé en 3 activités:
    
    #=======EDA(Exploratory Data Analysis)
La prémière c'est l'analyse et l'exploration de nos données ce qu'on appelle en anglais(Exploratory Data Analysis).
Ici l'objectif c'est de se mettre à l'aise avec le dataset de comprendre au maximum les différentes variables pour ensuite
définir une stratégie de modélisation. Qu'est ce qu'on va faire de nos données pour atteindre notre objectif.
Une fois qu'on a défini une stratégie qu'on a compris un petit peu ce qu'on va faire avec les données, on peut passer à la
2ème activité qui est le pre-processing.


    #=======Pre-processing
Pre-processing qui veut dire le pré-traitement des données .
Ici l'objectif c'est de transformer le dataset pour le mettre dans un format qui va etre propice au developpement des 
modèles de machine learning.Donc on va faire l'encodage, on va eliminer les valeurs manquantes, on va faire de la
 selection de variables etc...pour arriver au final à la 3ème activité qui l'activité de modélisation.
    
    
    #=======Modelling
Ici le but c'est de créer un modèle de machine learning de l'entrainer, de l'évaluer et de tenter de l'améliorer en 
sélectionnant un petit peu d'autre variable, en changant un petit peu ce qu'on a fait dans pre-processing, on va aussi
 peut etre comparé ce modèle avec la performance d'autres modèles de machine learning.
 Tout ca dans le but de l'objectif qu'on s'était fixé au tout debut.
    
"""

#========Exploratory Data Analysis===============

"""
Objectif comprendre au maximum les données dont on dispose pour définir une stratégie de modélisation

Checklist de base(non-exhaustive)

Analyse de la forme:
    
    -Identifiction de la target
    -Nombre de lignes et de colones
    -Types de variables
    -Identification des valeurs manquantes
    
Analyse du fond:
     -Visualisation de la target(Histogramme / Boxplot)
     -Compréhension des différentes variables (Internet)
     -Visualisation des relations features-target(Histogramme / Boxplot)
     -Identification des outliers
"""

#=================Pre-processing========================

"""
Objectif: transforme le data pour le mettre dans un format propice au Machine Learning.

Checklist de base(non-exhaustive):
    -Création du Train set / Test Set
    -Elimination des NaN: dropna(), imputation, colonnes<<vides>>
    -Encodage
    -Suppression des outliers néfastes au modèle
    -Feature Selection
    -Feature Engineering
    -Feature Scaling
    
"""  

#======================Modelling(Modélisation)============

"""
Objectif: développer un modèle de machine learning qui réponde à l'objectif final.

Checklist de base(non-exhaustive):
    -Définir une fonction d'évaluation
    -Entrainement de différents modèles
    -Optimisation avec GridSearchCV
    -(Optionnel) Analyse des erreurs et retour au Preprocessing / EDA
    -Learning Curve et prise de décision
    
"""
