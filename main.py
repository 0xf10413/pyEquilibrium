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
from learning_algorithm import LearningAlgorithm

#random.seed(1) # Fixer l'aléatoire

# Pick one !
#algorithm = DummyLearningAlgorithm() # For human players
algorithm = LearningAlgorithm() # For DDPG algo
board = Board()

application = Application(board, algorithm)
application.main_loop()
