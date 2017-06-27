#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sin, cos, pi, hypot, atan2

import numpy as np
import pygame as pg

from settings import (
        HIGH_BALL_RADIUS,
        LOW_BALL_ELASTICITY,
        LOW_BAR_SIZE,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        GRAVITY,
        WHITE,
        RED,
        LOW_BALL_RADIUS,
        WITH_SATURATE,
        )

import random as rd

"""
Divers objets utilisables sur le plateau de jeu
Convention utilisée pour représenter les points d'une barre
 une liste, dont les éléments décrivent la barre comme ceci
      1 ---- 2
      |      |
      4 ---- 3
"""

class GameOverException(BaseException):
    pass
class LowBallException(GameOverException):
    pass
class HighBallException(GameOverException):
    pass

class LowBar(object):
    """
    Barre inférieure du plateau
    Ne se déplace pas, peut seulement tourner
    """
    def __init__(self, x, y, w, h, theta=0):
        self.center = np.array([x,y])
        self.size = (w,h)
        self.points = [None, None, None, None]
        self.theta = theta
        self.updatePoints()

    def updatePoints(self):
        theta = self.theta
        rot_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        vec_x = rot_matrix.dot(np.array([self.size[0]/2,0]))
        vec_y = rot_matrix.dot(np.array([0, self.size[1]/2]))
        self.points[0] = self.center - vec_x - vec_y
        self.points[1] = self.center + vec_x - vec_y
        self.points[2] = self.center + vec_x + vec_y
        self.points[3] = self.center - vec_x + vec_y

    def rotate(self, d_theta, dt):
        self.theta += d_theta
        if WITH_SATURATE:
            self.theta = min(pi/2-.1, max(-pi/2+.1, self.theta))
        self.updatePoints()


class HighBar(object):
    """
    Barre supérieure du plateau
    Se déplace horizontalement
    """
    def __init__(self, x, y, w, h, pos=0):
        self.center = np.array([x,y])
        self.size = (w,h)
        self.points = [None, None, None, None]
        self.pos = pos
        self.updatePoints()

    def updatePoints(self):
        vec_x = np.array([self.size[0]/2, 0])
        vec_y = np.array([0, self.size[1]/2])
        self.points[0] = self.center - vec_x - vec_y
        self.points[1] = self.center + vec_x - vec_y
        self.points[2] = self.center + vec_x + vec_y
        self.points[3] = self.center - vec_x + vec_y

    def move(self, dx, dt):
        self.center[0] += dx
        if WITH_SATURATE:
            self.center[0] = max(self.size[0]/2,
                    min(WINDOW_WIDTH-self.size[0]/2, self.center[0]))
        self.updatePoints()


class LowBall(object):
    """
    Balle inférieure du plateau
    Se déplace suivant les lois de la physique standard, soumise à la gravité
    """
    def __init__(self, x, y, offset):
        self.surface = pg.Surface((LOW_BALL_RADIUS*2, LOW_BALL_RADIUS*2))
        self.center = np.array([x, y])
        self.prev_center = np.array([x, y])
        self.offset = offset

        self.contact = True
        self.hasToRebound = False

        self.radius = LOW_BALL_RADIUS
        self.speed = [0., 0.]
        pg.draw.circle(self.surface, RED, (int(self.radius), int(self.radius)), \
                int(self.radius))

    def get_rect(self, theta):
        return pg.Rect((int(self.center[0]-self.radius + self.offset*sin(theta)),\
                int(self.center[1]- self.radius - self.offset*cos(theta))), \
                (int(2*self.radius), int(2*self.radius)))

    def die(self):
        raise LowBallException("Perdu!")

    def move(self, theta, X_p, dt):
        if self.center[0] < 0 or self.center[0] > WINDOW_WIDTH or \
           self.center[1] > WINDOW_HEIGHT:
            self.die()
        if self.hasToRebound:
            self.hasToRebound = False
            #if hypot(self.center[0]-X_p[0], self.center[1] - X_p[1]) > low_bar_size[0]/2\
            #        + self.radius:
            if self.rebound(theta, X_p, dt):
                return

        X_prev = self.center
        X_pprev = self.prev_center
        A = np.matrix([[cos(theta), sin(theta)], [sin(theta), -cos(theta)]])
        B = np.matrix([[dt**2*GRAVITY*sin(theta) + \
            cos(theta)*(2*X_prev[0] - X_pprev[0]) + sin(theta)*(2*X_prev[1]-X_pprev[1])],\
            [X_p[0]*sin(theta) - X_p[1]*cos(theta)]])

        new_center = A.dot(B).flatten().tolist()[0]
        acc_x = new_center[0] + self.prev_center[0] - 2*self.center[0]
        acc_x /= dt**2
        acc_y = new_center[1] + self.prev_center[1] - 2*self.center[1]
        acc_y /= dt**2
        #print("Accélération:",acc_x, acc_y)
        if (acc_y > GRAVITY) or \
                hypot(self.center[0]-X_p[0], self.center[1] - X_p[1]) > LOW_BAR_SIZE[0]/2\
                + self.radius:
            if self.contact:
                #print("Décollage !")
                pass
            else:
                #print("Toujours décollée")
                pass
            self.contact = False
            self.move_without_contact(dt)
            return

        if not self.contact:
            self.contact = True
            #print("Atterissage !")
            self.hasToRebound = True
        else:
            self.speed[0] = (new_center[0] - self.center[0])/dt
            self.speed[1] = (new_center[1] - self.center[1])/dt

        self.prev_center = np.array(self.center).copy()
        self.center = new_center

    def move_without_contact(self, dt):
        new_center_y = GRAVITY*dt**2 + 2*self.center[1] - self.prev_center[1]
        new_center_x = 2*self.center[0] - self.prev_center[0]

        self.speed[0] = (new_center_x - self.center[0])/dt
        self.speed[1] = (new_center_y - self.center[1])/dt
        self.prev_center = np.array(self.center).copy()
        self.center = [new_center_x, new_center_y]

    def rebound(self, theta, X_p, dt):
        v_x, v_y = tuple(self.speed)
        norm_v = hypot(v_x, v_y)
        if LOW_BALL_ELASTICITY*norm_v < GRAVITY*dt:
            #print("Rebound cancelled")
            return False;

        angle_incidence = atan2(-v_y, -v_x) - (theta - pi/2)
        angle_refracte_absolu = theta - pi/2 - angle_incidence
        new_v_x = LOW_BALL_ELASTICITY * norm_v * cos(angle_refracte_absolu)
        new_v_y = LOW_BALL_ELASTICITY * norm_v * sin(angle_refracte_absolu)
        new_norm_v = hypot(new_v_x, new_v_y)

        #print("Rebound!")
        self.speed = [new_v_x, new_v_y]
        self.prev_center = np.array(self.center).copy()
        self.center[0] = self.center[0] + new_v_x*dt
        self.center[1] = self.center[1] + new_v_y*dt
        return True

class HighBall(object):
    """
    Balle supérieure du plateau
    Se déplace comme dans un casse-brique
    """
    def __init__(self, x, y):
        self.surface = pg.Surface((HIGH_BALL_RADIUS*2, HIGH_BALL_RADIUS*2))
        self.center = np.array([x, y])
        self.prev_center = np.array([x, y])

        self.radius = HIGH_BALL_RADIUS
        #self.speed = [random.uniform(.03, .1), random.uniform(-.1, .1)]
        self.speed = [0.02*rd.uniform(-1,1), .05*rd.uniform(-1,1)]
        pg.draw.circle(self.surface, WHITE, (int(self.radius), int(self.radius)), \
                int(self.radius))

    def get_rect(self):
        return pg.Rect((int(self.center[0]-self.radius),\
                int(self.center[1]- self.radius)), \
                (int(2*self.radius), int(2*self.radius)))

    def die(self):
        raise HighBallException("Perdu!")

    def move(self, bar, dt):
        self.prev_center = self.center.copy()
        self.center[0] += self.speed[0]*dt
        self.center[1] += self.speed[1]*dt
        if self.center[1] + self.radius > bar.center[1] - bar.size[1]/2:
            if self.center[0] - self.radius > bar.center[0] - bar.size[0]/2 and \
               self.center[0] + self.radius < bar.center[0] + bar.size[0]/2:
                   #print("Relaunch")
                   self.speed[1] = -abs(self.speed[1])
            elif self.center[1] - self.radius > bar.center[1] + bar.size[1]/2:
                self.die()

        if self.center[0] - self.radius < 0 or self.center[0] + self.radius > WINDOW_WIDTH:
            self.speed[0] = -self.speed[0]
        if self.center[1] - self.radius < 0 or self.center[1] + self.radius > WINDOW_HEIGHT:
            self.speed[1] = abs(self.speed[1])
