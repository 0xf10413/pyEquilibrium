#! /usr/bin/env python3

import sys
import random
from time import clock, sleep
from math import sin, cos, pi, hypot, atan2
import numpy as np
import pygame as pg
from pygame.locals import *

from genetics import GeneticAlgorithm
from application import Board, Application, DummyLearningAlgorithm

#random.seed(1) # Fixer l'aléatoire

# Pick one !
#algorithm = DummyLearningAlgorithm() # For human players
algorithm = GeneticAlgorithm() # For genetic algorithm
board = Board()

application = Application(board, algorithm)
application.main_loop()
