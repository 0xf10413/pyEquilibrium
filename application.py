#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import pi

import pygame as pg
from pygame.locals import DOUBLEBUF
import numpy as np

from board import Board
from learning_algorithm import LearningAlgorithm

from boarditems import HighBar, LowBar, HighBall, LowBall, GameOverException
from settings import (
        WINDOW_HEIGHT,
        WINDOW_WIDTH,
        WINDOW_SIZE,
        HIGH_BAR_SIZE,
        LOW_BAR_SIZE,
        LOW_BALL_RADIUS,
        WITH_FASTER,
        FPS,
        BLACK,
        WHITE,
        CYAN,
        )


class Application(object):
    """
    Classe générale d'application
    Se construit à partir d'un plateau de jeu et d'un algo d'apprentissage
    """
    def __init__(self, board, learning_algorithm):
        if not isinstance(learning_algorithm, LearningAlgorithm):
            raise ValueError("This is not a LearningAlgorithm : {}".
                    format(learning_algorithm))
        self.learning_algorithm = learning_algorithm
        self.board = board

        self.screen = pg.display.set_mode(WINDOW_SIZE, DOUBLEBUF)
        self.running = True
        self.clock = pg.time.Clock()
        pg.font.init()
        self.font = pg.font.Font(None, 36)
        self.opened = True

    def main_loop(self):
        while self.opened:
            # Mesure de l'avancement depuis le dernier tour de boucle
            # Bloquer si on est en avance
            delta_t = 0
            global WITH_FASTER
            if WITH_FASTER:
                delta_t = int(1000/FPS)
            else:
                delta_t = self.clock.tick(FPS)
                delta_t = int(1000/FPS)

            # Traiter les inputs
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.opened = False
                    continue
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_q:
                        self.opened = False
                        continue
                    elif event.key == pg.K_s:
                        WITH_FASTER = not WITH_FASTER
                        continue
                    elif event.key == pg.K_k:
                        WITHOUT_KEPT = not WITHOUT_KEPT
                        continue

            # Récupérer les prochaines entrées à donner à l'algorithme d'apprentissage
            X = self.board.fetch_state()

            # Récupérer l'avis de l'algorithme sur la situation
            reaction = 0
            text_to_print = ""
            if not isinstance(self.learning_algorithm, DummyLearningAlgorithm):
                reaction, text_to_print = self.learning_algorithm.act(X, delta_t)
            else:
                if pg.key.get_pressed()[pg.K_LEFT] != pg.key.get_pressed()[pg.K_RIGHT]:
                       if pg.key.get_pressed()[pg.K_LEFT]:
                           reaction = -10
                       elif pg.key.get_pressed()[pg.K_RIGHT]:
                           reaction = 10

            # Avancer la simulation d'un atome de temps
            # ou détecter un Game Over
            if not self.board.tick(reaction, delta_t):
                self.board = Board()
                self.learning_algorithm.inform_died()

            # Redessiner la scène
            text_to_print = self.font.render(text_to_print, 1, CYAN)
            self.board.redraw(self.screen, text_to_print)
            self.screen.blit(text_to_print, text_to_print.get_rect())

            pg.display.flip()


class DummyLearningAlgorithm(LearningAlgorithm):
    """
    Un algorithme d'apprentissage trivial, qui ne fait rien
    """
    def act(self, X, delta_t):
        return (0, "Player")

    def inform_died(self):
        pass