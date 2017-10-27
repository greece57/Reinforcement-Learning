""" Abstract Player """
class Player():
    """ This class should not be initialized. Inherit from this to create an AI """

    def __init__(self, name):
        """ Initialize Variables """
        self.name = name
        self.cards = []

        #inGame
        self.last_enemy_move = -1
        self.points = 0

        #afterGame
        self.last_game = 0.0

        #overAll
        self.won_games = 0
        self.lost_games = 0
        self.tied_games = 0

        self._init()

    def init_for_game(self, total_cards, points):
        """ Initialize Player for a new Game """
        self.cards = total_cards
        self.points = points

    def enemy_played(self, move):
        """ save enemys move """
        self.last_enemy_move = move

    def won_round(self):
        """ Called by the Game to inform about round Outcome
            You played a higher card then the other player. """
        self.points += 1
        self._won_round()

    def lost_round(self):
        """ Called by the Game to inform about negative round Outcome
            You played a lower card then the other player."""
        self._lost_round()

    def tie_round(self):
        """ Called by the Game to inform about equal round Outcome.
            Both players played the same card """
        self._tie_round()

    def perform_move(self):
        """ Called by the Game to ask for the next Move of the Player """
        move = self._choose_move()
        self.cards.remove(move)
        return move

    def won_game(self):
        """ Called by the Game to inform about positive Game Outcome """
        self.won_games += 1
        self.last_game = 1
        self._game_over()

    def lost_game(self):
        """ Called by the Game to inform about negative Game Outcome """
        self.lost_games += 1
        self.last_game = -1
        self._game_over()

    def tie_game(self):
        """ Called by the Game to inform about equal Game Outcome """
        self.tied_games += 1
        self.last_game = 0
        self._game_over()

    ### TO BE IMPLEMENTED ###
    def _init(self):
        pass

    def _choose_move(self):
        pass

    def _won_round(self):
        pass

    def _lost_round(self):
        pass

    def _tie_round(self):
        pass

    def _game_over(self):
        pass
