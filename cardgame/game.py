""" Game """
class Game:
    """
    This class shows a GameObject which is initiated with 2 players.
    It then is played turn by turn
    """

    def __init__(self):
        self.players = []
        self.verbose = False
        self.stats = []
        self.curr_round = 0

    def init_game(self, player1, player2, verbose=False):
        """
        Initialize the Game.
        Vebose prints out which player played which card and who won at the end
        """
        self.players = [player1, player2]
        self.verbose = verbose
        self.stats = []

    def play_round(self):
        """
        Simulate or Play on round of the Match.
        Both Players will be asked to perform a move by calling the perform_move function.
        Afterwards, depending on who won, either the won_round or the lost_round function
        will be called for each player.
        """
        move_p0 = self.players[0].perform_move()
        move_p1 = self.players[1].perform_move()

        self.players[0].enemy_played(move_p1)
        self.players[1].enemy_played(move_p0)

        if self.verbose:
            print(self.players[0].name + " played " + str(move_p0))
            print(self.players[1].name + " played " + str(move_p1))
            
        if move_p0 > move_p1:
            self.players[0].won_round()
            self.players[1].lost_round()
        elif move_p0 < move_p1:
            self.players[1].won_round()
            self.players[0].lost_round()
        else:
            self.players[0].tie_round()
            self.players[1].tie_round()

        self.curr_round += 1

    def start_new_match(self):
        """ Starts a new Match """

        self.players[0].cards = list(range(1, 11))
        self.players[1].cards = list(range(1, 11))
        self.players[0].points = 0
        self.players[1].points = 0
        self.curr_round = 0

        while self.curr_round < 10:
            self.play_round()

        if self.players[0].points > self.players[1].points:
            self.players[0].won_game()
            self.players[1].lost_game()
            self.stats.extend([0])
        elif self.players[0].points < self.players[1].points:
            self.players[0].lost_game()
            self.players[1].won_game()
            self.stats.extend([1])
        else:
            self.players[0].tie_game()
            self.players[1].tie_game()
            self.stats.extend([-1])
