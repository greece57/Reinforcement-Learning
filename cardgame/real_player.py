""" RealPlayer """
from cardgame.player import Player

class RealPlayer(Player):
    """ Implementing a real player """

    def _init(self):
        print("Choose your Name")
        self.name = input()

    def _choose_move(self):
        if self.last_enemy_move > -1:
            print("Enemy played " + str(self.last_enemy_move))

        move = -1
        while move not in self.cards:
            print("Which Card do you want to play?")
            print(self.cards)
            move = int(input())

        print("You played " + str(move))

        return move

    def _game_over(self):
        if self.last_game == 1:
            print("Yeay you won")
        elif self.last_game == 0:
            print("It's a tie")
        else:
            print("Ohh you lost")
