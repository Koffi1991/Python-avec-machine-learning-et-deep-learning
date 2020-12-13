"""
les fonctions les plus importantes, les plus utiles en data sciences,en deep learning
et en machine learning
"""

list_1 = [15,78,-96,3,4]
print("L'element le plus grand de ma liste est: ",max(list_1))  # elle me retourne l'element maxi de ma liste

print("L'element le plus grand de ma liste est: ",min(list_1))  # elle me retourne l'element mini de ma liste

print("la taille de ma liste est: ",len(list_1)) # elle me retourne la taille de ma liste

print("la somme de tous les elements de ma liste est: ",sum(list_1)) # elle retourne la somme de tous les elements de ma liste

list_2 = [True,True,False]
print("",all(list_2)) # elle retourne false cas tous les elements de liste ne sont pas egaux à True

print("",any(list_2)) # elle retourne True cas ya au moins un element qui est egal à True

print("",all(list_1)) # elle retourne false cas tous les elements de liste ne sont pas True
print("",any(list_1)) # elle retourne false cas tous les elements de liste ne sont pas True


# les fonctions qui sont extrenement utiles
"""
elles permenttent de convertir un type de variable à une autre
par exemple : une liste en tuple ou un tuple à une liste etc...
"""
# exemple

x = 20
v = str(x)      # convertir un entier en un string(chaine de caractère)
print(type(v))
print(v)

y = "13"      
f = int(y)    # convertir un string en int(entier)
print(f)

g = 12
j = float(g)
print(j)

# exemple sur les listes, tuples etc...en gros tuple(), liste() sont des fonctions

list_3 = [12,89,46,24,69,21]

print(tuple(list_3))                 # convertir une liste en tuple

tuple_1 = (12,89,46,24,69,21)
print(list(tuple_1))                # convertir un tuple en liste

inventaire = {
              "Bananes" : 548,
              "Pommes" :  963,
              "Poires" : 216
              }

print(list(inventaire.values()))
print(list(inventaire.keys()))

# la fonction input()
print("\n")
b = input("Veuillez svp entrer le numero: ")   # b ici est un string
print("vous avez choisire:",b)

# la fonction input() pour pouvoir faire des calcules avec 
print("\n")             
b = int(input("Veuillez svp entrer le numero: "))  # on covertir b en int
d = b + 6
print("vous avez choisire:",d)


"""
la fonction format permet de personnaliser des chaines de 
caracteres, de les rendre dynamique 
"""
# exemple
 
h = 25
ville = "Paris"
message = "la temperature est de {} degC à {}".format(h,ville)
print(message)

l = 30
ville_2 = "Lyon"
msg = f"la temperature est de {l} degC à {ville_2}"
print(msg)
print("\n")

"""
L'une des raison pour laquelle on utilise la fonction format()
c'est pour acceder à certains clef de notre dictionnaire
dans une boucle FOR par exemple

Donc imaginons que là j'ai cree un dictionnaire parametre
pour un reseau de neurones bah avec la technique format, je peux
acceder au different couche de mon neurone 1,2, etc...
à travers une boucle une boucle en combinant par exemple
la lettre W puis les accolades et la valeur qu'a la valeur i 
a chaque un instant de la boucle FOR
"""


import numpy as np

parametre = {
              "W1" : np.random.randn(2,4),
              "b1" : np.zeros((2,1)),
              "W2" : np.random.randn(2,2),
              "b2" : np.zeros((2,1))
              
             }
    
for i in range(1,3):
    print("couche",1)
    print(parametre["W{}".format(i)])
    
print("\n")  
"""
La fonction open() qui nous permet d'ouvrir des fichers ou d'ecrire dans les fichiers
ou de creer des fichers et de les enregitrer sous un certain nombre
sur notre ordinateur
"""

f = open("fichier.txt","w")   # pour ecrir dans un fichier
f.write("Bonjour")
f.close()

f = open("fichier.txt","r")   # pour lire dans un fichier
print(f.read())


# une autre technique d'utilisation de open()

with open("fichier.txt","r") as f:
    print(f.read())
    
"""
un petit algorithm qui nous ecrit dans un fichier texte tous les nombre carrés
de i allant de 0 à 10
"""
with open("fichier.txt","w") as f:
    for i in range(10):
        f.write("{}^2 = {}\n".format(i, i**2))
   
with open("fichier.txt","r") as f:
    print(f.read())





