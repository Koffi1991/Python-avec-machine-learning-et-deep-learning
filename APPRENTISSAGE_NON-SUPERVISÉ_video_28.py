# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:14:08 2020

@author: Administrator
"""
"""
L'apprentissage Non-Supervisé (Unsupervised Learning) est une technique de Machine Learning tres populaire.
 Dans ce tutoriel Python sur sklearn en français, je vous dévoile les algorithmes les plus importants :
     -K-Means Clustering, 
     -IsolationForest, 
     -PCA (Analyse en composantes principales)


Pour rappel, l’apprentissage supervisé est une technique d’apprentissage qui consiste à montrer à la machine
 des exemples X, y de ce qu’elle doit apprendre.
À l’inverse, l’apprentissage non-supervisé consiste à fournir à la machine uniquement des données X, et
 lui demander d’analyser la structure de ces données pour apprendre elle-même à réaliser certaines tâches.

1. Clustering
Une des applications les plus populaires de l’apprentissage non-supervisé est le Clustering. Le principe 
est de laisser la machine apprendre à trier des données selon leur ressemblances (et donc en analysant 
uniquement les features X).
Les algorithmes à connaitre :
- K-Means : Tres rapide, mais non-efficace sur les clusters non-convexes.
- AgglomerativeClustering : assez lent, mais efficace sur les données non-convexes
- DBSCAN : efficace sur les données non-convexes, mais sélection du nombre de clusters automatique
Applications :
- Trier des documents, des photos, des tweets
- Segmenter la clientèle d’une entreprise
- Optimiser l’organisation d’un système informatique, etc…

2. Détection d’Anomalies
Un autre exemple d’application de l’apprentissage non-supervisé est la Détection d’Anomalies. En analysant
 la structure X des données, la machine est capable de trouver les échantillons dont les features sont tres 
 éloignées de celles des autres échantillons. Ces échantillons sont alors considérés comme étant des 
 anomalies.
Les algorithmes à connaitres :
- IsolationForest : Efficace pour détecter des outliers dans le train_set
- Local Outlier Factor : Efficace pour détecter des anomalies futures
Applications :
- Nettoyer un Dataset des valeurs aberrantes qui le composent
- Détecter un comportement anormal sur un site Internet ou sur une caméra de surveillance
- Maintenance prédictive des machines d’une usine

3. Réduction de dimension
La dernière application très importante de l’apprentissage non-supervisé est la réduction de dimension.
 Le principe est de réduire la complexité superflue d’un dataset en projetant ses données dans un espace
 de plus petite dimension (un espace avec moins de variables). Le but est d’Accélérer l’apprentissage 
 de la machine et de Lutter contre le fléau de la dimension.
Algorithmes a connaitres :
- Analyse en composantes principales (PCA) : le plus populaire et le plus simple a comprendre
- TSNE
- Isomap

Applications :
- Visualisation de données : afficher sur un graphique 2D un espace de grande dimension
- Compression de dataset : réduire au maximum le poids d’un dataset en conservant un maximum de qualité

"""

#========APPRENTISSAGE NON-SUPERVISE==========================

"""
Qu'est ce que c'est l'apprentissage non-supersé?
C'est une méthode d'apprentissage dans laquelle au lieu de montrer à la machine des exemple X/Y de ce qu'elle
doit apprendre, on lui fournit uniquement des données X et on lui demande d'analyser la structure de ces données
pour apprendre à elle mm à réaliser certaines taches. 
Par exemple la machine peut apprendre à classer des données en les regroupant uniquement selon leur ressemblance
c'est ce qu'on appelle faire du clustering ou en francais de la classification non-supervisé. Avec cette technique
on peut faire enormement de choses comme classer des documents, des photos, classer des tweets, segmenter la clientèle
d'une entreprise.On verra comment faire tout ca avec :
    -l'aglo de K-Means clustering.

Une autre tache que la machine peut grace à l'apprentissage non-supervisé est la detection d'anomalies. La machine
analyse la structure de nos données et parvient à trouver les échantillons dont les caracteristiques sont très éloignées 
de celles des autres échantillons et  ca ca nous permet de developper des systèms de securité, de détection de fraude
bancaire, de détection de défaillance dans une usine.
On verra comment developper tels système avec :
    -l'aglo Isolation Forest.
    
Une 3ème application populaire de l'apprentissage non-surpervisé est la réduction de la dimensionnalité.
En fait en analysant la structure de nos données, la machine est capable d'apprendre comment simplifier cette structure
tout en conservant les principales informations.
Exemple c'est comme si la machine étudie la structure d'un dessin et apprend à redessiner ce dessin de facon plus 
simple tout en conservant les principaux élements. D'un point de vue math, la machine apprend en fait à projeter nos données
dans des espaces de plus petites dimensions.
On verra comment faire cela avec:
    -l'algo de Analyse en Composantes Principales(PCA)
    
Les applications de cet algo sont absolument genial. On peut non seulement simplifier la complexité superflu que pourrait
avoir un dataset ce qui facilite alors bcp l'apprentissage de la machine pour des problèmes de régression ou de classifications
mais on peut également visualiser en 2 dimensions ou en 3 dimensions des espaces qui vont bien au delà de notre imagination.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#====================K-Means CLUSTERING=======================

"""
Ici le principe est de laisser la machine apprendre à classer nos données selon leur ressemblance pour ca on va utiliser:
    -l'algo de K-Means Clustering.
Comment cet algo fonctionne?
Imaginez qu'on désire regrouper des données en 3 clusters. Pour ca, on va commencer à placer 3 points appelés K centroid au
hasard dans le dataset.(Ce sont les barycentres des futures clusters).
Ensuite on affecte chaque point du dataset au centroid le plus proche. Ce qui nous donne 3 clusters puis on deplace chaque
chaque centroid au milieu de son cluster.
Ensuite on recommence, on affecte chaque point de notre dataset au centroid le plus proche puis on deplace chaque centoid 
au centre de son cluster et on va continuer ainsi jsuqu'à ce que les centroids convergent vers une position d'équilibre.
Donc voilà les 2 étapes de l'algo de K-Means clustering
L'algo de K-Means clusterin est en fait un algo itératif qui fonctionne en 2 étapes:
    -on affecte les points du dataset au centroid le plus proche
    -On calcule la moyenne de chaque cluster et on y déplace le centroid.
En résumé:
L'algo de K-Means cherche la position des centres qui minimise la distance entre les points d'un cluster(Xi) et le centre(U(mur)j)
 de ce dernier   

Note: Cela équivaut à minimiser la variance des clusters
"""

"""
Hyper-paramètre:
    -n_clusters : nombre K de clusters
    -n_init: nombre d'exécution(10)
    -max_iter: nombre d'itérations(300)
    -Init: type d'initialisation(k-means++)
    
Méthodes:
    -Fit(X): exécute l'algorithme K-Means
    -Predict(X): centroid le plus proche de X
    -Score(X): calcul de l'inertia(négatif)
    
Attributs:
    -cluster_centers_: position des centroids
    -Labels: équivalent de Predict(Xtrain)
    -Inertai_: calcul de l'inertia(positif)
"""
"""
pour faire ca, on importe l'estimateur KMeans depuis le module cluster
Ensuite on cée un modèle dans lequel on précise le nombre de cluters qu'on veut avoir par exemple 3
on peut également definir le nombre d'initialisation que l'on veut avoir pour notre algo c'est 
d'ailleurs le nombre de fois que l'algo devrais s'exécuter de base ce nombre est fixé à 10 on a 
pas besoin d'y toucher.
on peut également définir le nombre d'itérations max avec max_inter de base il est fixé à 200 ou 300
pareil on a pas besoin d'y toucher. E pour finir un dernier parametre init c'est à dire quelle 
stratégie d'initialisation nous allons utiliser pour notre algo par defaut c'est la methode
K-Means++ qui est utilisé cette technique consite à placer nos centroids sur des points très 
éloigné de notre dataset ceci dans le but d'accéler la convergence de nos centroids vers
des positions déquilibre. c'est une très bonne méthode d'initialisation donc pas besoin d'y toucher

En gros quand on utilise l'algo de K-Means tout ce qu'on a à faire c'est d'écrire n_clusters et de 
choisir un nombre de clusters.
"""
from sklearn.cluster import KMeans

# Génération de données
X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.4, random_state=0)
plt.scatter(X[:,0], X[:,1])

plt.figure()

model = KMeans(n_clusters=3)
model.fit(X)
# on peut voir comment sont classés nos différents échantillons
print("nos différents échantillons\n",model.labels_)

print("\nOn peut également utiliser la methode predict pour voir nos différents échantillons\n",model.predict(X))

"""
On peut visualiser nos données
c= pour voir les couleurs
"""
plt.scatter(X[:,0], X[:,1], c = model.predict(X))

# on peut afficher la position finale de nos centroids
print("\nle tableau de nos centroids dont 3 centroids et 2 variables\n", model.cluster_centers_)

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='r')
print("\nLe score est\n",model.score(X))

"""
Dc c'est grace à ces 3 centroids qu'on va pourvoir faire de futurs prédictions si jamais
on a un nouvel échantillon qui arrive dans nos données, pour savoir à quel cluster
il appartient, on va tout simplement calculer sa distance avec les 3 centroids et la 
distance la plus proche ce sera le centroid qu'il appartiendra
"""
"""
On peut calculer aussi l'inertia qui va nous donner la somme des distances entre les point
d'un cluster et le centroid
ici on a 30.870531280140668 donc ca depend des unité qu'on utilise dans notre dataset euro etc...
"""
print("Inertia\n",model.inertia_)

#===============Elbow Method
"""
comment etre nous sure de choisir le bon nombre de clusters pour notre algo ?
Bah pour trouver le bon nombre de clusters:
    -Elbow Method: Détecter une zone de coude dans la minimisation du cout(inertia_)
    elle consiste à tracer l'évolution du cout de notre modèle en fonction du nombre de clusters
    et de détecter dans ce graphique une zone de coude. Cette zone nous indique le nombre de 
    cluster optimal c'est à dire celui qui nous permet de rédure au maximum le cout de notre modèle
    tout en conservant un nombre raisonnable de clusters
    
"""
plt.figure()
inertia = []
# on definit la rangée de valeur qu'on veut tester
K_range = range(1, 20)

for k in K_range:
    model = KMeans(n_clusters=k).fit(X)
    inertia.append(model.inertia_)

plt.plot(K_range, inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele (Inertia)')


#===========Isolation Anomaly Detection==============
#============L'algorithme Isolation Forest
"""
L'idée est la suivante:
On effectue une série de splits aléatoires, et on compte le nombre de splits qu'il faut effectuer
pour pouvoir isoler nos échantillons.
Plus ce nombre est petit et plus il y a de chance qu'un échantillon soit en fait une anomalie.
Exemple:
    Prenons un dataset et effectuons au hasard un premier split c'est à dire une première découpe 
    pour ca on commence par choisir une de nos variable au hasard par exemple X1 et on trace quelque
    part au hasard un trait orthogonale à X1 ensuite on vérifie si ce trait nous a permis d'isoler un de
    nos échantillons si oui on s'arret sinon on continue on choisit une de nos variable au hasard par exemple
    X2 pui on trace quelque au hasard un trait orthoganale à X2 rt on verifie si ce trait nous a permis d'isoler
    un échantillon sinon non on continue encore avec la mm procedure jusq'à isoler un echantillon.
    
Pour eviter cela il existe une solution. C'est de demander à la machine de générer plusieurs estimateur qui
vont chacun effectué une séquence de split aléatoire en considerant alors l'ensemble de leurs résultats nous
somme capable de disqualifier les quelques petites erreus qui pourrait etre commises par certains estimateur
c'est ce qu'on appelle une technique d'ensemble parce qu'on entraine plusieurs estimateurs pour ensuite
considérer l'ensemble de leurs estimation.

En gros l'algo isolation forest est en fait un ensemble de générateurs de type arbre d'où son nom foret puisque
c'est un ensemble d'arbres donc une foret et les splits que l'on effectue represente en fait les embranchements différents
arbre de notre foret
"""

plt.figure()

X, y = make_blobs(n_samples=50, centers=1, cluster_std=0.1, random_state=0)
X[-1,:] = np.array([2.25, 5])

plt.scatter(X[:,0], X[:, 1])

plt.figure()


from sklearn.ensemble import IsolationForest
"""
 Ici je dis à mon modele qu'il 1% de dechets dans mon dataset
est ce qu'il peut les identifier
On peut utiliser cette technique pour détecter tout type d'anomalie
qu ca soit des fraudes bancaires, des défaillances techniques dans une usine, un comportement anormal sur une
caméra de surveillance ou tout simplement les outliers qui pourrait y avoir dans un datasetqu'on desire nettoyer

"""
model = IsolationForest(contamination=0.01)
model.fit(X)

# pour identifier notre anomalie
plt.scatter(X[:,0], X[:, 1], c=model.predict(X))

"""
 tout simplement les outliers qui pourrait y avoir dans un datasetqu'on desire nettoyer et c'est qu'on va faire
 en trouvant dans le dataset digit quels sont les chiffres moins bien  écrit ce qu'il est préférable d'éliminer
 avant de donner notre dataset à la machine pour une tache de classification
"""
plt.figure()

from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
X = digits.data
y = digits.target

print("\nla dimension de notre dataset 1797 images de 64 pixels\n",X.shape)

plt.imshow(images[42])

"""
Ce qu'on va faire c'est de nettoyer  ce dataset des images qui serait mal ecrite
"""

plt.figure()

model = IsolationForest(random_state=0, contamination=0.02)
model.fit(X)

"""
les 1 representent les bonnes données et les -1 represents les 
données qui ont des anomalies
c'est pourquoi dans notre prediction on a uniquement des 1 donc on va juste avoir
2% de -1
"""
print("\on affichit notre predict\n", model.predict(X))

"""
si on veut filtrer tout ca et afficher tout ca, on va faire du boolean indexing
"""
# une variable qui contiendra tout les predictions qui sont = -1
outliers = model.predict(X) == -1 
"""
si on regarde le contenu de outlier on aura que des falses, il aura
true là où on aura des -1
"""
print("\c'est quoi qui est dans la variable outliers\n",outliers)

"""
avec ce tableau outliers, on peut l'injecter dans xy ou bien image
"""
print("\nLes images pour lesquels on a des outliers\n",images[outliers])

plt.figure()
# pour afficher une de ces images par exemple l'image 0
plt.imshow(images[outliers][0])

# pour ce que c'est cette image 
plt.title(y[outliers][0])

plt.figure()

plt.figure(figsize=(12, 3))
for i in range(10):
  plt.subplot(1, 10, i+1)
  plt.imshow(images[outliers][i])
  plt.title(y[outliers][i])
  
  
#====================PCA La Réduction de Dimension==========

"""
Le principe est de projeter nos données sur des axes appelés composantes principales,
en cherchant à minimiser la distance entre nos points et leur projections.
De cette manière, on réduit la dimension de notre dataset, tout en préservant au 
maximum la variance de nos données et c'est ca le plus important dans l'analyse en composant 
principeles

Dc pour faire de la reduction de dimension, il suffit de charger le transformer PCA depuis
le module decomposition, de preciser le nombre de dimensions sur lesquels on desire
projeter nos données et transformer nos données avec la methode fit_transform()

Mtn la question c'est comment choisir le nombre de composantes sur lesquels projeter
nos données?
 Il y a 2 cas possibles:
     -visualisation de données:
         on projette notre dataset dans un espace 2D ou 3D (n_components =2 ou 3)
     - Compression de données: (regression ou classification)
         Réduire au maximum la taille du dataset tout en conservant 95-99%
         de la variance de nos données
"""
#=============== La visualisation 2D de nos données

plt.figure()

from sklearn.decomposition import PCA

model = PCA(n_components=2)

print("\non transforme nos données\n",model.fit_transform(X)) 
print("\nles dimensions apres la transformation\n",model.fit_transform(X).shape)

PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)

x_pca = model.transform(X)
plt.scatter(x_pca[:,0], x_pca[:,1], c=y)

plt.colorbar()

""" 
pour savoir à quoi correspondre les axes les abscisse et les ordonnées,
il faut analyser le contenu de chaque composante

"""
print("\n Les differents contenu\n", model.components_)

"""
chaque composant contient 64 valeurs et si nous avons 64 valeurs c'est parce
qu'en fait chaque composante est une combinaison linéaire des 64 variables de notre
dataset
"""
print("\n Si on analyse les dimensions\n", model.components_.shape)

#=====================Compression de données
print("\nCompression de données\n")
"""
Pour on va commencer par entrainer notre modèle PCA sur le mm nombre de dimensions que
l'on a dans X c'est à dire 64.
Ensuite nous allon examiner quel est le pourcentage de variance préservé par chacune
de nos composantes pour ca on ecrit exeplained_variance_ratio ce qui va nous donner le 
pourcentage des variance qui est préservée par chacune des composantes donc la composante
1,2,3 etc...et on va faire la somme cumulée de toutes ses variances ou pourcentages et là
de 0.14% à 100% et ce qu'on va faire c'est de trouver le moment ou on atteint 95% ou 99%
"""
print(X.shape)

#model = PCA(n_components=64)

#model = PCA(n_components=40)

#model = PCA(n_components=3)

"""
on peut directement ecrire dans n_component le pourcentage qu'on desire avoir
"""
model = PCA(n_components=0.95)


X_reduced = model.fit_transform(X)
print("\nle pourcentage de variance préservé par chacune de nos composante\n",model.explained_variance_ratio_)

print("\nla somme cumulée de toutes ses variances ou pourcentages\n",np.cumsum(model.explained_variance_ratio_))

plt.figure()

plt.plot(np.cumsum(model.explained_variance_ratio_))

"""
pour trouver l'endroit où c'est superieur à par exemple 99
On trouve 40 donc c'est à dire c'est à partir de la 40eme composante principale que l'on atteint 99% de la variance
de notre modèle.
A partir il ne reste plus qu'à réentrainer notre modèle PCA avec le notre qu'on a trouvé dc 40 et ensuite on est 
sure d'avoir reduire notre dimension à la meilleure valeur tout en gardant 99% de l'information de notre dataset
"""
print("\nl'endroit où c'est superieur à par exemple 99\n",np.argmax(np.cumsum(model.explained_variance_ratio_)>0.99))

"""
Si on veut observer à quoi ressemble ces images une fois quelles ont été compressés, il faut commencer par les 
decompresser pour qu'elles aient à nouveau 64 pixel
"""
X_recovered = model.inverse_transform(X_reduced)

plt.figure()

#redimensionner pour 8 pixels hauter et largeur
plt.imshow(X_recovered[0].reshape((8,8)))

"""
Si on veut savoir combien de compsantes sont utilisés dans notre modèle
"""
print("\ncombien de compsantes sont utilisés dans notre modèle\n",model.n_components_)



# Exemple 2

plt.figure()

n_dims = X.shape[1]
model = PCA(n_components=n_dims)
model.fit(X)

variances = model.explained_variance_ratio_

meilleur_dims = np.argmax(np.cumsum(variances) > 0.90)


plt.bar(range(n_dims), np.cumsum(variances))
plt.hlines(0.90, 0, meilleur_dims, colors='r')
plt.vlines(meilleur_dims, 0, 0.90, colors='r')


#Exemple 3

plt.figure()

model = PCA(n_components=0.99)
print("\n Exemple 3\n",model.fit(X))

X_compress = model.fit_transform(X)
X_decompress = model.inverse_transform(X_compress)

plt.subplot(1, 2, 1)
plt.imshow(X[0,:].reshape((8,8)), cmap='gray')
plt.title('originel')
plt.subplot(1, 2, 2)
plt.imshow(X_decompress[0,:].reshape((8,8)), cmap='gray')
plt.title('Compressé')
