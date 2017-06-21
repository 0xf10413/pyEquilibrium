#! /usr/bin/env python3

import sys
import random
from time import clock, sleep
from math import sin, cos, pi, hypot, atan2
import numpy as np
import pygame as pg
from pygame.locals import *

from neural import Population

### Options
WITH_SATURATE = True
WITH_FASTER = False
WITHOUT_KEPT = False # Should we test the kept people anyway ?

size = width, height = 320, 400
high_ball_radius = 5
low_ball_radius = 10
low_bar_size = low_bar_w, low_bar_h = int(2*width/3), 4
high_bar_size = high_bar_w, high_bar_h = int(width/6), 4
low_ball_elasticity = .5 # <1, >0
gravity = .001

#random.seed(1) # Fixer l'aléatoire

class GameOverException(BaseException):
    pass
class LowBallException(GameOverException):
    pass
class HighBallException(GameOverException):
    pass

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
cyan = (0, 255, 255)
FPS = 30


screen = pg.display.set_mode(size, DOUBLEBUF)
opened = True
pg.font.init()
font = pg.font.Font(None, 36)
score = 0



clock = pg.time.Clock()

# 1 2
# 4 3
class LowBar(object):
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
            self.center[0] = max(self.size[0]/2, min(width-self.size[0]/2, self.center[0]))
        self.updatePoints()


class LowBall(object):
    def __init__(self, x, y, offset):
        self.surface = pg.Surface((low_ball_radius*2, low_ball_radius*2))
        self.center = [x, y]
        self.prev_center = [x, y]
        self.offset = offset

        self.contact = True
        self.hasToRebound = False

        self.radius = low_ball_radius
        self.speed = [0., 0.]
        pg.draw.circle(self.surface, red, (int(self.radius), int(self.radius)), \
                int(self.radius))

    def get_rect(self, theta):
        return pg.Rect((int(self.center[0]-self.radius + self.offset*sin(theta)),\
                int(self.center[1]- self.radius - self.offset*cos(theta))), \
                (int(2*self.radius), int(2*self.radius)))

    def die(self):
        raise LowBallException("Perdu!")

    def move(self, theta, X_p, dt):
        if self.center[0] < 0 or self.center[0] > width or self.center[1] > height:
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
        B = np.matrix([[dt**2*gravity*sin(theta) + \
            cos(theta)*(2*X_prev[0] - X_pprev[0]) + sin(theta)*(2*X_prev[1]-X_pprev[1])],\
            [X_p[0]*sin(theta) - X_p[1]*cos(theta)]])

        new_center = A.dot(B).flatten().tolist()[0]
        acc_x = new_center[0] + self.prev_center[0] - 2*self.center[0]
        acc_x /= dt**2
        acc_y = new_center[1] + self.prev_center[1] - 2*self.center[1]
        acc_y /= dt**2
        #print("Accélération:",acc_x, acc_y)
        if (acc_y > gravity) or \
                hypot(self.center[0]-X_p[0], self.center[1] - X_p[1]) > low_bar_size[0]/2\
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

        self.prev_center = self.center.copy()
        self.center = new_center

    def move_without_contact(self, dt):
        new_center_y = gravity*dt**2 + 2*self.center[1] - self.prev_center[1]
        new_center_x = 2*self.center[0] - self.prev_center[0]

        self.speed[0] = (new_center_x - self.center[0])/dt
        self.speed[1] = (new_center_y - self.center[1])/dt
        self.prev_center = self.center.copy()
        self.center = [new_center_x, new_center_y]

    def rebound(self, theta, X_p, dt):
        v_x, v_y = tuple(self.speed)
        norm_v = hypot(v_x, v_y)
        if low_ball_elasticity*norm_v < gravity*dt:
            #print("Rebound cancelled")
            return False;

        angle_incidence = atan2(-v_y, -v_x) - (theta - pi/2)
        angle_refracte_absolu = theta - pi/2 - angle_incidence
        new_v_x = low_ball_elasticity * norm_v * cos(angle_refracte_absolu)
        new_v_y = low_ball_elasticity * norm_v * sin(angle_refracte_absolu)
        new_norm_v = hypot(new_v_x, new_v_y)

        #print("Rebound!")
        self.speed = [new_v_x, new_v_y]
        self.prev_center = self.center.copy()
        self.center[0] = self.center[0] + new_v_x*dt
        self.center[1] = self.center[1] + new_v_y*dt
        return True

class HighBall(object):
    def __init__(self, x, y):
        self.surface = pg.Surface((high_ball_radius*2, high_ball_radius*2))
        self.center = [x, y]
        self.prev_center = [x, y]

        self.radius = high_ball_radius
        #self.speed = [random.uniform(.03, .1), random.uniform(-.1, .1)]
        self.speed = [.12, .05]
        pg.draw.circle(self.surface, white, (int(self.radius), int(self.radius)), \
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

        if self.center[0] - self.radius < 0 or self.center[0] + self.radius > width:
            self.speed[0] = -self.speed[0]
        if self.center[1] - self.radius < 0 or self.center[1] + self.radius > height:
            self.speed[1] = abs(self.speed[1])


population = Population()
for player in population.mega_iterator():
    try:
        if WITHOUT_KEPT and player.origin == "Kept":
            score = player.fitness
            continue
        high_bar = HighBar(width/2, height/4, high_bar_size[0], high_bar_size[1])
        high_ball = HighBall(width/2, height/5)
        low_bar = LowBar(width/2, 3*height/4, low_bar_size[0], low_bar_size[1])
        low_ball = LowBall(width/2-low_ball_radius,3*height/4-low_ball_radius-low_bar.size[1], \
                low_ball_radius + low_bar_size[1]/2)
        clock = pg.time.Clock()
        while opened:
            delta_t = 0
            if WITH_FASTER:
                delta_t = int(1000/FPS)
            else:
                delta_t = clock.tick(FPS)
                delta_t = int(1000/FPS)
            score += delta_t

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    opened = False
                    continue
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_q:
                        opened = False
                        continue
                    elif event.key == pg.K_s:
                        WITH_FASTER = not WITH_FASTER
                        continue
                    elif event.key == pg.K_k:
                        WITHOUT_KEPT = not WITHOUT_KEPT
                        continue

            #if bool(pg.key.get_pressed()[pg.K_LEFT]) != bool(pg.key.get_pressed()[pg.K_RIGHT]):
            #    if pg.key.get_pressed()[pg.K_LEFT]:
            #        low_bar.rotate(.1, delta_t)
            #        high_bar.move(-width/30, delta_t)
            #    elif pg.key.get_pressed()[pg.K_RIGHT]:
            #        low_bar.rotate(-.1, delta_t)
            #        high_bar.move(width/30, delta_t)



            text_score = font.render(player.origin + str(score), 1, cyan)

            # Récupération des variables d'entrée du réseau
            low_ball_x = low_ball.center[0]
            low_ball_y = low_ball.center[1]
            low_ball_vx = low_ball.speed[0]
            low_ball_vy = low_ball.speed[1]

            high_ball_x = high_ball.center[0]
            high_ball_y = high_ball.center[1]
            high_ball_vx = high_ball.speed[0]
            high_ball_vy = high_ball.speed[1]

            high_bar_x = high_bar.center[0]
            high_bar_y = high_bar.center[1]
            low_bar_x = low_bar.center[0]
            low_bar_y = low_bar.center[1]

            theta = low_bar.theta
            X = np.array([high_ball_x - high_bar_x,  \
                    100*max(high_ball_vy, 0)*(high_ball_x - high_bar_x)/ \
                        (high_ball_y - high_bar_y - 6)**3, \
                    high_ball_vx, high_ball_vy,\
                    low_ball_x - low_bar_x, low_ball_y - low_bar_y, \
                    low_ball_vx, low_ball_vy, theta])

            reaction = player.ACT(X)
            low_ball.move(low_bar.theta, low_bar.center, delta_t)
            high_ball.move(high_bar,delta_t)

            #if reaction == "Left":
            #    low_bar.rotate(.2, delta_t)
            #    high_bar.move(-width/15, delta_t)
            #elif reaction == "Right":
            #    low_bar.rotate(-.2, delta_t)
            #    high_bar.move(width/15, delta_t)
            #elif reaction == "Nothing":
            #    pass
            low_bar.rotate(reaction*pi/width, delta_t)
            high_bar.move(reaction, delta_t)

            screen.fill(black)
            pg.draw.polygon(screen, white, low_bar.points)
            pg.draw.polygon(screen, white, high_bar.points)
            screen.blit(high_ball.surface, high_ball.get_rect())
            screen.blit(low_ball.surface, low_ball.get_rect(low_bar.theta))
            screen.blit(text_score, text_score.get_rect())


            pg.display.flip()


    except LowBallException:
        print("La balle inférieure est tombée…")
        if not WITH_FASTER:
            sleep(.5)
    except HighBallException:
        print("La balle supérieure est tombée…")
        if not WITH_FASTER:
            sleep(.5)
    finally:
        player.die(score)
        score = 0
