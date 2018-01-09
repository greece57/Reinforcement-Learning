""" Reinforcement Learning Based Player """
import random
import numpy as np
from cardgame.player import Player

class RLAgent(Player):
    """ This Player tried to learn over time """

    def __init__(self, name, hidden_layers, learning_rate, update_frequency):
        """ Define Variables """

        # My own
        self.enemy_cards = []
        self.running_reward = 0
        self.ep_history = []
        self.last_action = -1
        self.last_card = -1
        self.current_state = [[], []]

        super().__init__(name)

    def _init(self):

        # Game Infos
        total_cards = 10

        self._start_new_game()

    def _choose_move(self):
        self.last_card = random.choice(self.cards)

        return self.last_card

    def _won_round(self):
        self._update_game(won=True)

    def _lost_round(self):
        self._update_game(lost=True)

    def _tie_round(self):
        self._update_game(tie=True)

    def _game_over(self):
        self._start_new_game()

    def _start_new_game(self):
        self.enemy_cards = list(range(1, 11))
        self.cards = list(range(1, 11))
        self.current_state = self._create_state()
        self.running_reward = 0
        self.ep_history = []

    def _update_game(self, won=False, lost=False, tie=False, wrong_card=False, reward_extra=0):
        if self.last_enemy_move in self.enemy_cards:
            self.enemy_cards.remove(self.last_enemy_move)

    def _create_state(self):
        return np.append([1 if card in self.cards else 0 for card in range(1, 11)],
                         [1 if card in self.enemy_cards else 0 for card in range(1, 11)])

    def finalize(self):
        pass

def discount_rewards(rewards):
    """ take 1D float array of rewards and comopute discounted reward """
    gamma = 0.99

    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for idx in reversed(range(0, rewards.size)):
        running_add = running_add * gamma + rewards[idx]
        discounted_r[idx] = running_add
    return discounted_r
