#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# import pour DQL
import numpy as np
import random

import timeit

class LearningAlgorithm(object):
    pass

class DDPGAlgorithm(LearningAlgorithm):
    """
    Un algorithme d'apprentissage trivial, qui ne fait rien
    """
    def act(self, X, delta_t):
        return (0, "Player")

    def inform_died(self):
        pass


    """
    Interface représentant un algorithme d'apprentissage
    """
    def __init__(self):
        from ReplayBuffer import ReplayBuffer
        from ActorNetwork import ActorNetwork
        from CriticNetwork import CriticNetwork
        import tensorflow as tf
        from keras.models import model_from_json, Model
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation, Flatten
        from keras.engine.topology import Merge
        from keras.engine.training import collect_trainable_weights
        from keras.layers.convolutional import Convolution2D
        from keras.layers import Permute, merge
        from keras.optimizers import Adam

        from OU import OU
        self.OU = OU()       #Ornstein-Uhlenbeck Process

        # definition des constantes de l'experience
        self.BUFFER_SIZE = 300000 #taille du buffer
        self.BATCH_SIZE = 64 #taille du batch
        self.GAMMA = 0.95  # prise en compte du futur, parametre classique
        self.TAU = 0.001     #Target Network HyperParameters
        self.LRA = 0.0001    #Learning rate for Actor
        self.LRC = 0.001     #Lerning rate for Critic
        self.action_dim = 1
        self.state_dim = 9,  #shape of sensors input
        self.EXPLORE = 100000
        self.episodes_number = 1000 #nombre d'episode de la simulation
        self.steps_per_episode = 1000 #nombre de mvt avant passage au nouvel episode
        self.epsilon = 1     #initial epsilon
        self.step = 0
        self.time_since_gameover = 0
        self.total_reward = 0
        self.episode = 0
        # 'administrativ' variables
        self.train_indicator = 1
        self.done = False

        # Tensorflow GPU optimization
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        from keras import backend as K
        K.set_session(self.sess)

        # networks definitions
        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.buff = ReplayBuffer(self.BUFFER_SIZE)    #Create replay buffer

        self.gameover = False

    def act(self, X, delta_t):
        a_t = np.zeros([1,self.action_dim])
        noise_t = np.zeros([1,self.action_dim])

        a_t_original = self.actor.model.predict(np.array([X]), verbose=0) # to change

        noise_t[0][0] = self.train_indicator * max(self.epsilon, 0) * self.OU.function(a_t_original[0][0], 0, 0.15, 0.20)

        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

        if self.step == self.steps_per_episode-1:
            self.done = True
            self.step = 0
        else:
            self.done = False
            self.step += 1

        r_t = self.reward()
        self.buff.add(X, a_t[0], r_t, X, self.done) #Add replay buffer
        # initialisation des variables d'apprentissage propores à chaque tour
        loss = 0
        self.epsilon -= 1.0 / self.EXPLORE

        self.time_since_gameover += 1

        # noise à changer en fonction des inputs, un peu du pif

        # if not isinstance(self, DummyLearningAlgorithm):
        #     reaction, text_to_print = self.act(X, delta_t)
        # else:
        #     if pg.key.get_pressed()[pg.K_LEFT] != pg.key.get_pressed()[pg.K_RIGHT]:
        #            if pg.key.get_pressed()[pg.K_LEFT]:
        #                reaction = 10
        #            elif pg.key.get_pressed()[pg.K_RIGHT]:
        #                reaction = -10
        reaction = a_t[0]

        batch = self.buff.getBatch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.zeros((states.shape[0],1))

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.GAMMA*target_q_values[k]

        loss += self.critic.model.train_on_batch([states,actions], y_t)
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()

        self.total_reward += r_t

        text_to_print = "Episode" + str(self.episode) + "Step" + str(self.step) + "Action" + str(a_t) + "Reward" + str(r_t) + "Loss" + str(loss)

        self.gameover = False

        return reaction, text_to_print

    def inform_died(self):
        """
        Indique à l'algorithme qu'un GameOver a été atteint
        """
        self.gameover = True
        self.time_since_gameover = 0
        self.episode += 1
        self.step = 0

    def reward(self):
        if self.gameover:
            return -1
        else:
            return self.time_since_gameover/self.steps_per_episode