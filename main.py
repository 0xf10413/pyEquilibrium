#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random
from time import clock, sleep
from math import sin, cos, pi, hypot, atan2
import numpy as np
import pygame as pg
from pygame.locals import *

from application import Application
from board import Board
from application import DummyLearningAlgorithm
from genetics import GeneticAlgorithm
from learning_algorithm import DDPGAlgorithm

random.seed(1) # Fixer l'aléatoire

# Pick one !
#algorithm = DummyLearningAlgorithm() # For human players
algorithm = DDPGAlgorithm() # For DDPG algo
#algorithm = GeneticAlgorithm() # For genetic algo
board = Board()

application = Application(board, algorithm)
application.main_loop()
