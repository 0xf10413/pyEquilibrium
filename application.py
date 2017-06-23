#!/usr/bin/env python3
from math import pi

import pygame as pg
from pygame.locals import DOUBLEBUF
import numpy as np

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
                           reaction = 10
                       elif pg.key.get_pressed()[pg.K_RIGHT]:
                           reaction = -10

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


class Board(object):
    """
    Classe représentant un plateau de jeu
    """
    def __init__(self):
        self.high_bar = HighBar(WINDOW_WIDTH/2, WINDOW_HEIGHT/4,
                HIGH_BAR_SIZE[0], HIGH_BAR_SIZE[1])
        self.high_ball = HighBall(WINDOW_WIDTH/2, WINDOW_HEIGHT/5)
        self.low_bar = LowBar(WINDOW_WIDTH/2, 3*WINDOW_HEIGHT/4,
                LOW_BAR_SIZE[0], LOW_BAR_SIZE[1])
        self.low_ball = LowBall(WINDOW_WIDTH/2-LOW_BALL_RADIUS,
                3*WINDOW_HEIGHT/4-LOW_BALL_RADIUS-LOW_BAR_SIZE[1],
                LOW_BALL_RADIUS + LOW_BAR_SIZE[1]/2)

    def fetch_state(self):
        low_ball_x = self.low_ball.center[0]
        low_ball_y = self.low_ball.center[1]
        low_ball_vx = self.low_ball.speed[0]
        low_ball_vy = self.low_ball.speed[1]

        high_ball_x = self.high_ball.center[0]
        high_ball_y = self.high_ball.center[1]
        high_ball_vx = self.high_ball.speed[0]
        high_ball_vy = self.high_ball.speed[1]

        high_bar_x = self.high_bar.center[0]
        high_bar_y = self.high_bar.center[1]
        low_bar_x = self.low_bar.center[0]
        low_bar_y = self.low_bar.center[1]

        theta = self.low_bar.theta
        X = np.array([high_ball_x - high_bar_x,  \
                100*max(high_ball_vy, 0)*(high_ball_x - high_bar_x)/ \
                    (high_ball_y - high_bar_y - 6)**3, \
                high_ball_vx, high_ball_vy,\
                low_ball_x - low_bar_x, low_ball_y - low_bar_y, \
                low_ball_vx, low_ball_vy, theta])
        return X


    def tick(self, reaction, delta_t):
        """
        Met à jour le plateau de jeu selon l'input reaction, sachant qu'il s'est
        écoulé delta_t depuis le dernier appel
        """
        try:
            self.low_bar.rotate(reaction*pi/WINDOW_WIDTH, delta_t)
            self.high_bar.move(reaction, delta_t)
            self.low_ball.move(self.low_bar.theta, self.low_bar.center, delta_t)
            self.high_ball.move(self.high_bar, delta_t)
        except GameOverException:
            return False
        return True

    def redraw(self, screen, text_to_print):
        screen.fill(BLACK)
        pg.draw.polygon(screen, WHITE, self.low_bar.points)
        pg.draw.polygon(screen, WHITE, self.high_bar.points)
        screen.blit(self.high_ball.surface, self.high_ball.get_rect())
        screen.blit(self.low_ball.surface, self.low_ball.get_rect(self.low_bar.theta))

class LearningAlgorithm(object):
    """
    Interface représentant un algorithme d'apprentissage
    """
    def act(self, X, delta_t):
        """
        Effectue une itération d'apprentissage, étant donné l'entrée X,
        sachant qu'il s'est écoulé delta_t depuis le dernier appel

        Valeur de retour attendue : un coupe (r, text) où r est l'output
        à injecter dans le plateau de jeu, et text un texte à afficher
        sur la fenêtre
        """
        raise NotImplementedError("Calling abstract method")

    def inform_died(self):
        """
        Indique à l'algorithme qu'un GameOver a été atteint
        """
        raise NotImplementedError("Calling abstract method")

class DummyLearningAlgorithm(LearningAlgorithm):
    """
    Un algorithme d'apprentissage trivial, qui ne fait rien
    """
    def act(self, X, delta_t):
        return (0, "Player")

    def inform_died(self):
        pass
