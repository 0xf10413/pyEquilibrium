#!/usr/bin/env python3
import numpy as np
import operator
import copy
from random import uniform,choice

from game.application import LearningAlgorithm

from settings import (
        POPULATION_SIZE,
        NUM_PARAMETERS,
        RAND_MIN,
        RAND_MAX,
        MAX_GENERATION,
        REF_VECTOR,
        MUTATION_EPS,
        )

class Individual(object):
    def __init__(self, coeffs=None):
        self.fitness = 0
        self.origin = "None"
        if coeffs is None:
            self.coeffs = np.array([uniform(RAND_MIN, RAND_MAX) \
                                    for i in range(NUM_PARAMETERS)])
        else:
            selfs.coeffs = coeffs.copy()

    def __str__(self):
        return str(self.coeffs) + " w/ fitness " + str(self.fitness)

    def mutate(self):
        result = Individual()
        for i in range(self.coeffs.size):
            result.coeffs[i] = self.coeffs[i] + MUTATION_EPS*uniform(RAND_MIN, RAND_MAX)
        return result


    def crossover(self, other):
        result = Individual()
        result.coeffs = self.fitness*self.coeffs + other.fitness*other.coeffs
        result.coeffs = 1/(self.fitness + other.fitness) * result.coeffs
        return result

    def ACT(self, inputs, delta_t):
        act = np.dot(self.coeffs, inputs)
        self.fitness += delta_t
        return act

class GeneticAlgorithm(LearningAlgorithm):
    def __init__(self):
        self.people = [Individual() for i in range(POPULATION_SIZE)]
        # uncomment below to get the best player we found
        #self.people[0].coeffs = np.array(REF_VECTOR.copy())
        self.generation = 1
        self.current_people_index = 0

    def inform_died(self):
        """
        Si cette fonction est appelée, c'est que le jeu vient de se solder par un gameover
        Le plateau va se réinitialiser, l'algorithme doit réagir
        """
        self.current_people_index += 1

    def act(self, X, delta_t):
        """
        One tick for the algorithm
        Tries to advance the current candidate, or else goes to the next one
        """
        if self.generation > MAX_GENERATION:
            print("The end !")
            print("Best player:", self.best, "(", self.best.fitness,")")
            return 0,"The end"

        if self.current_people_index < POPULATION_SIZE:
            player = self.people[self.current_people_index]
            text_to_print = player.origin + str(player.fitness)
            return player.ACT(X, delta_t), text_to_print
        else:
            self.current_people_index = 0
            self.next()
            print("=======================")
            print(" End of generation ", self.generation)
            print("=======================")
            return self.act(X, delta_t)

    def next(self):
        """
        Creates the following population
        """
        ONE_QUARTER = POPULATION_SIZE//4
        THREE_QUARTER = 3*POPULATION_SIZE//4
        ONE_HALF = POPULATION_SIZE//2

        # Sorting w/ respect to fitness
        self.people.sort(key=operator.attrgetter('fitness'), reverse=True)
        self.best = self.people[0]

        # First quartile is kept
        next_population = []
        for i in self.people[:ONE_QUARTER]:
            next_population.append(copy.deepcopy(i))
            next_population[-1].origin = "Kept"
        print("Kept", len(self.people[:ONE_QUARTER]), "from last gen")

        # Second quartile is made from crossover with first
        for i in range(ONE_QUARTER, ONE_HALF):
            next_population.append(self.people[i].crossover(self.people[i-ONE_QUARTER]))
            next_population[-1].origin = "Crossed"
        print("Crossover'ed ", len(range(ONE_HALF, THREE_QUARTER)), " from last gen")

        # Third quartile is mutated
        for i in range(ONE_HALF, THREE_QUARTER):
            next_population.append(self.people[i-ONE_HALF].mutate())
            next_population[-1].origin = "Mutated"
        print("Mutated ", len(range(ONE_QUARTER,ONE_HALF)), " from last gen")

        # The rest is filled with new items
        i = len(next_population)
        while i < len(self.people):
            next_population.append(Individual())
            #copy.deepcopy(choice(self.people[:ONE_QUARTER])))
            next_population[-1].origin = "New"
            i += 1
        print("Filled to", i)

        assert len(self.people) == len(next_population)

        # Last print of the population
        for i in range(len(self.people)):
            print("#",i,":", self.people[i].fitness)
        print("Avg = ", np.average([p.fitness for p in self.people]))
        print("Max = ", self.people[0].fitness, " ; min = ", self.people[-1].fitness)
        self.people = next_population
        self.generation += 1
