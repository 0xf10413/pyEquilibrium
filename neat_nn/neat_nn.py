#!/usr/bin/env python3
import numpy as np
import operator
import copy
from random import uniform,choice

import neat

from game.application import LearningAlgorithm

class NEATAlgorithm(LearningAlgorithm):
    def __init__(self):
        # Configuration
        self.config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                'neat_nn/config-feedforward',
                )
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(False))
        self.current_idx = 1
        self.current_gen = 0
        self.best = None

        for genome in self.population.population:
            self.population.population[genome].fitness = 0


    def inform_died(self):
        ## TODO: faire plusieurs itérations sur le même élément
        ## Pour s'assurer qu'il n'a pas juste eu une balle "facile"
        self.current_idx += 1

    def act(self, X, delta_t):
        # Passage à la génération suivante si besoin
        if self.current_idx > len(self.population.population):
            print("====================")
            print("End of generation #" + str(self.current_gen))
            print("====================")
            self.current_idx = 1
            self.current_gen += 1
            print("Best was with {} : {}".format(self.best.fitness, self.best))
            self.best = None
            self.population = neat.Population(self.config,
                    initial_state=(
                        self.population.population,
                        self.population.species,
                        self.current_gen)
                    )
            for genome in self.population.population:
                self.population.population[genome].fitness = 0
            return self.act(X, delta_t)

        genome = self.population.population[self.current_idx]
        genome.fitness += delta_t
        if self.best is None or self.best.fitness < genome.fitness:
            self.best = genome

        # Construction du NN correspondant pour calculer la réponse
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        return (net.activate(X)[0], "Gen{},#{},{}".format(
            self.current_gen,
            self.current_idx,
            genome.fitness))
