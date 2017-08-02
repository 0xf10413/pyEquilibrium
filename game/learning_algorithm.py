#!/usr/bin/env python3
class LearningAlgorithm(object):
    """
    Classe de base pour les algorithmes d'apprentissage
    """
    pass

class DummyLearningAlgorithm(LearningAlgorithm):
    """
    Un algorithme d'apprentissage trivial, qui ne fait rien
    """
    def act(self, X, delta_t):
        return (0, "Player")

    def inform_died(self):
        pass
