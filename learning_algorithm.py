#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

# import pour DQL
import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.engine.topology import Merge
from keras.engine.training import collect_trainable_weights
from keras.layers.convolutional import Convolution2D
from keras.layers import Permute, merge
from keras.optimizers import Adam
import tensorflow as tf


from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit





class LearningAlgorithm(object):
    """
    Interface représentant un algorithme d'apprentissage
    """
    def __init__(self):

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
            self.time_since_gameover = 0
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

    # def act(self, X, delta_t):
    #     """
    #     Effectue une itération d'apprentissage, étant donné l'entrée X,
    #     sachant qu'il s'est écoulé delta_t depuis le dernier appel
    #
    #     Valeur de retour attendue : un coupe (r, text) où r est l'output
    #     à injecter dans le plateau de jeu, et text un texte à afficher
    #     sur la fenêtre
    #     """
    #     raise NotImplementedError("Calling abstract method")

    def inform_died(self):
        """
        Indique à l'algorithme qu'un GameOver a été atteint
        """
        self.gameover = True
        self.time_since_gameover = 0

    def reward(self):
        """
        Calcul la reward d'une action dans un état
        """
        if self.gameover:
            return -1
        else:
            return self.time_since_gameover/self.steps_per_episode

class DummyLearningAlgorithm(LearningAlgorithm):
    """
    Un algorithme d'apprentissage trivial, qui ne fait rien
    """
    def act(self, X, delta_t):
        return (0, "Player")

    def inform_died(self):
        pass
