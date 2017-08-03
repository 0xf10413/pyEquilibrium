#!/usr/bin/env python3

import sys
import random
from time import clock, sleep
from math import sin, cos, pi, hypot, atan2
import numpy as np
import pygame as pg
from pygame.locals import *

from game.application import Application
from game.board import Board
from game.learning_algorithm import DummyLearningAlgorithm
from genetics.genetics import GeneticAlgorithm
from ddpg.Algorithm import DDPGAlgorithm
from neat_nn.neat_nn import NEATAlgorithm

random.seed(1) # Fixer l'aléatoire

# Pick one !
#algorithm = DummyLearningAlgorithm() # For human players
#algorithm = GeneticAlgorithm() # For genetic algo
#algorithm = DDPGAlgorithm() # For DDPG algo
algorithm = NEATAlgorithm() # For NEAT algo
board = Board()

application = Application(board, algorithm)
application.main_loop()
