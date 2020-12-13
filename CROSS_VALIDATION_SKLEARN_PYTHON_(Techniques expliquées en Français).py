# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 04:52:09 2020

@author: Administrator
"""

"""
 les techniques de Cross-Validation, très utiles en Machine Learning, et je vous montre comment les mettre 
 en place dans Sklearn (Python).

Les principales techniques de Cross-Validations sont:

1) KFold
2) Leave One Out
3) ShuffleSplit
4) StratifiedKFold
5) GroupKFold

Pour les utiliser dans Python avec Sklearn, il faut les importer depuis le module sklearn.model_selection.

Par exemple:
from sklearn.model_selection import KFold

cv = KFold(n_splits=5)
cross_val_score(model, X, y, cv=cv)

"""


#=============================KFold Cross-Validation============
"""
1) KFold Cross-Validation :
Consiste à mélanger le Dataset, puis à le découper en K parties égales (K-Fold). Par exemple si le Dataset
 contient 100 échantillons et que K=5, alors nous aurons 5 paquets de 20 échantillons. Ensuite, la machine 
 s'entraine sur 4 paquet, puis s'évalue sur le paquet restant, et alterne les différentes combinaisons de
 paquet possibles. Au Final, elle effectue donc un nombre K d'entrainement (5 entraînements dans cette 
 situation).
Cette technique est LARGEMENT UTILISÉE, mais elle a un léger désavantage: si le dataset est hétérogène et 
comprend des classes déséquilibrées, alors il se peut que certain splits de Cross-Validation ne contiennent
 pas les classes minoritaires. Par exemple, si un dataset de 100 échantillons contient seulement
 10 échantillons de la classe 0, et 90 échantillons de la classe 1, alors il est possible que sur 5 Folds, 
 certains ne contiennent pas d'échantillon de la Classe 0.
"""

print("\nKFold Cross-Validation\n")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

print(x.shape)

plt.scatter(x[:,0], x[:,1], c = y, alpha = 0.8)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# on indique le nombre de split par exemple : 5 et random_state = 0 est l'etat 
# aleatoire sur lequel on veut etre au moment de melanger notre dataset avant la decoupe

cv = KFold(5, random_state=0)

# on va passer nos données x et y dans la fonction crosse_val_score()
print(cross_val_score(KNeighborsClassifier() ,x,y, cv = cv))



#============Leave One Out Cross Validation==========
"""
2) Leave One Out Cross Validation.
Cette technique est un cas particulier du K-Fold. En fait, il s'agit du cas ou K = "nombre d'échantillons 
du Dataset". Par exemple, si un Dataset contient 100 échantillons, Alors K =100. La machine s’entraîne donc
 sur 99 échantillons et s'évalue sur le dernier. Elle procède ainsi à 100 entraînements 
 (sur les 100 combinaisons possibles) ce qui peut prendre un temps considérable à la machine.
Cette technique est DÉCONSEILLÉE.
"""

print("\n")

print("Leave One Out Cross Validation\n")

from sklearn.model_selection import LeaveOneOut

cv = LeaveOneOut()

# on va passer nos données x et y dans la fonction crosse_val_score()
print(cross_val_score(KNeighborsClassifier() ,x,y, cv = cv))


#===============ShuffleSplit Cross-Validation========
"""
3) ShuffleSplit Cross-Validation :
Cette technique consiste à mélanger puis découper le Dataset en deux parties : Une partie de Train, et
 une partie de Test. Une fois l'entrainement puis l'évaluation complétée, On rassemble nos données, on les
 remélange, puis on redécoupe le DataSet dans les même proportions que précédemment. On répète ainsi l'action
 pour autant d'itérations de Cross-Validation que l'on désire. On peut ainsi retrouver plusieurs fois 
 les mêmes données dans le jeu de validation a travers les Itérations.
Cette technique est une BONNE ALTERNATIVE au K-FOLD, mais elle présente le même désavantage: si les classes 
sont déséquilibrées, alors on risque de manquer d'informations dans le jeu de Validation !
"""
from sklearn.model_selection import ShuffleSplit

print("\nLa fonction ShuffleSplit Cross-Validation\n")

# on definit le nombre de split qu'on veut avoir par exemple : 4
# Ensuit on definit une quantité qu'on veut avoir dans notre testsize
cv = ShuffleSplit(4, test_size = 0.2)

# on va passer nos données x et y dans la fonction crosse_val_score()
print(cross_val_score(KNeighborsClassifier() ,x,y, cv = cv))


#===============STRATIFIED K-FOLD====================
"""
4) STRATIFIED K-FOLD
Cette technique est un choix par défaut (mais consomme un peu plus de ressource que le K-FOLD). 
Elle consiste à mélanger le dataset, puis laisser la machine trier les données en "Strata" 
(c'est à dire en différentes classes) avant de former un nombre K de paquets (K-Fold) qui contiennent 
tous un peu de données de chaque Strata (de chaque Classe).
"""
from sklearn.model_selection import StratifiedKFold

print("\nLa fonction STRATIFIED K-FOLD Cross-Validation\n")

# on definit le nombre de split qu'on veut avoir par exemple : 4
# Ensuit on definit une quantité qu'on veut avoir dans notre testsize
cv = StratifiedKFold(4)

# on va passer nos données x et y dans la fonction crosse_val_score()
print(cross_val_score(KNeighborsClassifier() ,x,y, cv = cv))


#==================GROUP K-FOLD======================

"""
5) GROUP K-FOLD
Cette technique de Cross-Validation est TRÈS IMPORTANTE A CONNAITRE !
En Data Science, on fait souvent l’hypothèse que nos données sont indépendantes et tirées de la même
 distribution. Par exemple, les appartements d'un DataSet de l'immobiliers sont tous indépendants 
 (les uns des autres) et identiquement distribués.
Mais ce n'est pas toujours le cas ! Par exemple, les données d'un Dataset médical peuvent dépendre les 
unes des autres : si des gens d'une même famille sont diagnostiqué d'un cancer, alors le facteur génétique 
crée une dépendance entre les différentes données. Il faut donc Découper le Dataset en Groupe d'influence, 
c'est pour ca qu'il existe GROUP K-FOLD.
GroupKfold(5).split(X, y, groups)
"""
from sklearn.model_selection import GroupKFold

print("\nLa fonction GROUP K-FOLD Cross-Validation\n")

# on definit le nombre de split qu'on veut avoir par exemple : 4
# Ensuit on definit une quantité qu'on veut avoir dans notre testsize
cv = GroupKFold(5).get_n_splits(x,y, groups = x[:,0])

# on va passer nos données x et y dans la fonction crosse_val_score()
print(cross_val_score(KNeighborsClassifier() ,x,y, cv = cv))


