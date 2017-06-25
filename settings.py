#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Options générales pour le programme
"""

##############################
## Options d'algo génétique ##
##############################

# Algo génétique : accélération de l'algorithme en évitant le rendu à < 25fps ?
WITH_FASTER = False

# Algo génétique : teste-t-on les éléments conservés de la génération précédente ?
WITHOUT_KEPT = False

# Saturation des barres : les empêche-t-on de se déplacer à l'infini ?
WITH_SATURATE = True

# Paramètres génériques des algo génétiques
RAND_MIN = -25
RAND_MAX = 25

MUTATION_EPS = 1
POPULATION_SIZE = 50
MAX_GENERATION = 30
NUM_PARAMETERS = 9

# Vecteur de référence, "champion" dans certaines conditions initiales
REF_VECTOR = [ 1,  -7,  -6,  -1, -20,   2, -13,  -2, -10]

#######################
## Options de dessin ##
#######################

# Fenêtre
WINDOW_SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = 320, 400

# Balle du haut
HIGH_BALL_RADIUS = 5

# Balle du bas
LOW_BALL_RADIUS = 10
LOW_BALL_ELASTICITY = .5 # Note : dans )0,1(

# Barre du haut
HIGH_BAR_SIZE = HIGH_BAR_W, HIGH_BAR_H = int(WINDOW_WIDTH/6), 4

# Barre du bas
LOW_BAR_SIZE = LOW_BAR_W, LOW_BAR_H = int(2*WINDOW_WIDTH/3), 4

# Paramètres généraux de simulation
GRAVITY = .001
FPS = 30

##############
## Couleurs ##
##############
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
