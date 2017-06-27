#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

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

        # pour la prise en compte du passé
        self.states_list = []

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
        X = np.array([low_ball_x/WINDOW_WIDTH,
                      low_ball_y/WINDOW_HEIGHT,
                      low_ball_vx,
                      low_ball_vy,

                      high_ball_x/WINDOW_WIDTH,
                      high_ball_y/WINDOW_HEIGHT,
                      high_ball_vx,
                      high_ball_vy,

                      theta/pi
                     ])
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
