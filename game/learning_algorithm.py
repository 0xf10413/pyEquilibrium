#!/usr/bin/env python3
class LearningAlgorithm(object):
    """
    Classe de base pour les algorithmes d'apprentissage
    """
    def act(self, X, delta_t):
        """
        Un tick de l'algorithme, étant donné l'input X et l'écart de temps
        depuis le dernier appel delta_t
        """
        raise NotImplementedError("Did not redefine " + self.act.__name__)

    def inform_died(self):
        """
        Si cette fonction est appelée, c'est que le jeu vient de se solder par un gameover
        Le plateau va se réinitialiser, l'algorithme doit réagir
        """
        raise NotImplementedError("Dit not redefine " + self.inform_died.__name__)

class DummyLearningAlgorithm(LearningAlgorithm):
    """
    Un algorithme d'apprentissage trivial, qui ne fait rien.
    Prévu pour laisser le joueur prendre le contrôle
    """
    def act(self, X, delta_t):
        return (0, "Player")

    def inform_died(self):
        pass
