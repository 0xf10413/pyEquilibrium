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

from OU import OU
OU = OU()       #Ornstein-Uhlenbeck Process


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
        self.total_reward = 0


    def main_loop(self):

        for episode in range(self.learning_algorithm.episodes_number):

            print("Episode : " + str(episode) + " Replay Buffer " + str(self.learning_algorithm.buff.count()))

            # reset de l'environnement
            self.board = Board()
            self.learning_algorithm.time_since_gameover = 0
            self.learning_algorithm.gameover = False
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
            s_t = self.board.fetch_state()


            for step in range(self.learning_algorithm.steps_per_episode):


                # initialisation des variables d'apprentissage propores à chaque tour
                loss = 0
                self.learning_algorithm.epsilon -= 1.0 / self.learning_algorithm.EXPLORE
                a_t = np.zeros([1,self.learning_algorithm.action_dim])
                noise_t = np.zeros([1,self.learning_algorithm.action_dim])

                a_t_original = self.learning_algorithm.actor.model.predict(np.array([s_t]), verbose=0) # to change

                noise_t[0][0] = self.learning_algorithm.train_indicator * max(self.learning_algorithm.epsilon, 0) * OU.function(a_t_original[0][0], 0, 0.15, 0.20)

                a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

                # noise à changer en fonction des inputs, un peu du pif

                # if not isinstance(self.learning_algorithm, DummyLearningAlgorithm):
                #     reaction, text_to_print = self.learning_algorithm.act(X, delta_t)
                # else:
                #     if pg.key.get_pressed()[pg.K_LEFT] != pg.key.get_pressed()[pg.K_RIGHT]:
                #            if pg.key.get_pressed()[pg.K_LEFT]:
                #                reaction = 10
                #            elif pg.key.get_pressed()[pg.K_RIGHT]:
                #                reaction = -10
                reaction = a_t[0]

                # Avancer la simulation d'un atome de temps
                # ou détecter un Game Over
                if not self.board.tick(reaction, delta_t):
                    self.board = Board()
                    self.learning_algorithm.inform_died()


                self.learning_algorithm.time_since_gameover += 1
                s_t1 = self.board.fetch_state()
                r_t = self.learning_algorithm.reward()

                if step == self.learning_algorithm.steps_per_episode-1:
                    self.learning_algorithm.done = True
                else:
                    self.learning_algorithm.done = False

                self.learning_algorithm.buff.add(s_t, a_t[0], r_t, s_t1, self.learning_algorithm.done) #Add replay buffer

                # do the batch update
                batch = self.learning_algorithm.buff.getBatch(self.learning_algorithm.BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.zeros((states.shape[0],1))

                target_q_values = self.learning_algorithm.critic.target_model.predict([new_states, self.learning_algorithm.actor.target_model.predict(new_states)])

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + self.learning_algorithm.GAMMA*target_q_values[k]

                loss += self.learning_algorithm.critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = self.learning_algorithm.actor.model.predict(states)
                grads = self.learning_algorithm.critic.gradients(states, a_for_grad)
                self.learning_algorithm.actor.train(states, grads)
                self.learning_algorithm.actor.target_train()
                self.learning_algorithm.critic.target_train()

                self.total_reward += r_t
                s_t = s_t1.copy()

                print("Episode", episode, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

                self.learning_algorithm.gameover = False

                # Redessiner la scène
                text_to_print = "feature coming soon"
                text_to_print = self.font.render(text_to_print, 1, CYAN)
                self.board.redraw(self.screen, text_to_print)
                self.screen.blit(text_to_print, text_to_print.get_rect())

                pg.display.flip()

            print("TOTAL REWARD @ " + str(episode) +"-th Episode  : Reward " + str(self.total_reward))
            print("Total Step: " + str(step))
            print("")
