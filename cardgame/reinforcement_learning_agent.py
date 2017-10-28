""" Reinforcement Learning Based Player """
import numpy as np
from numpy.random import choice
import tensorflow as tf
from tensorflow.contrib import slim
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
        self.current_state = [[],[]]

        # Tensorflow
        self._hidden_layers = hidden_layers
        self._learning_rate = learning_rate
        self._update_frequency = update_frequency
        super().__init__(name)

    def _init(self):

        # Game Infos
        total_cards = 10

        # Tensorflow config
        self.state_in = tf.placeholder(shape=[None, 20], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, self._hidden_layers,
                                      biases_initializer=None, activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())
        self.output = slim.fully_connected(hidden, total_cards, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Training Proceedure
        # Feed the reward and chosen action into the network to compute loss,
        # and use that to update the network.

        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

        tf.reset_default_graph()

        init = tf.global_variables_initializer()
        self._sess = tf.Session()
        self._sess.run(init)

        self.grad_buffer = self._sess.run(tf.trainable_variables())
        for ix, grad in enumerate(self.grad_buffer):
            self.grad_buffer[ix] = grad * 0

        self._start_new_game()

    def _choose_move(self):
        a_dist = self._sess.run(self.output, feed_dict={self.state_in:[self.current_state]})
        self.last_action = choice(a_dist[0], p=a_dist[0]) # np.random.choice <- pylint hates me
        self.last_action = np.argmax(a_dist == self.last_action)

        if self.last_action not in self.cards:
            self._update_game(wrong_card=True)
            return self._choose_move(self)
        else:
            return self.last_action

    def _won_round(self):
        self._update_game(won=True)

    def _lost_round(self):
        self._update_game(lost=True)

    def _tie_round(self):
        self._update_game(tie=True)

    def _game_over(self):
        #Update the network
        self.ep_history = np.array(self.ep_history)
        self.ep_history[:, 2] = discount_rewards(self.ep_history[:, 2])
        feed_dict = {self.reward_holder:self.ep_history[:, 2],
                   self.action_holder:self.ep_history[:, 1],
                   self.state_in:np.vstack(self.ep_history[:, 0])}
        grads = self._sess.run(self.gradients, feed_dict=feed_dict)
        for idx, grad in enumerate(grads):
            self.grad_buffer[idx] += grad

        total_games = self.won_games + self.lost_games + self.tied_games
        if total_games % self._update_frequency == 0 and total_games != 0:
            feed_dict = dictionary = dict(zip(self.gradient_holders, self.grad_buffer))
            _ = self._sess.run(self.update_batch, feed_dict=feed_dict)
            for ix, grad in enumerate(grads):
                self.grad_buffer[ix] = grad * 0

        self._start_new_game()

    def _start_new_game(self):
        self.enemy_cards = list(range(1, 11))
        self.cards = list(range(1,11))
        self.current_state = self._create_state()
        self.running_reward = 0
        self.ep_history = []

    def _update_game(self, won=False, lost=False, tie=False, wrong_card=False):
        self.enemy_cards.remove(self.last_enemy_move)
        reward = int(won) + int(lost) * -1 + int(tie) * 0 + int(wrong_card) * -100 # False == 0, True == 1
        last_state = self.current_state
        self.current_state = self._create_state()
        self.ep_history.append([last_state, self.last_action, reward, self.current_state])

    def _create_state(self):
        return np.append([1 if card in self.cards else 0 for card in range(1, 11)],
                         [1 if card in self.enemy_cards else 0 for card in range(1, 11)])

def discount_rewards(r):
    """ take 1D float array of rewards and comopute discounted reward """
    gamma = 0.99

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r