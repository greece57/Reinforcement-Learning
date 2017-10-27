""" Random AI """
import random
from cardgame.player import Player

class RandomAIPlayer(Player):
    """ Returning random playable Cards """

    def _init(self):
        return 0

    def _choose_move(self):
        return random.choice(self.cards)

    def _game_over(self):
        return 0
