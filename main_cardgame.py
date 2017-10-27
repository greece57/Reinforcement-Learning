""" Let the Game begin """
from cardgame.random_ai import RandomAIPlayer
from cardgame.real_player import RealPlayer
from cardgame.game import Game

#p1 = RealPlayer("Niko")

p2 = RandomAIPlayer ("John")
p3 = RandomAIPlayer ("Peter")

total_games = 1000000

g = Game()
g.init_game(p2, p3)
for i in range(total_games):
    g.start_new_match()
    if i % (total_games / 10) == 0:
        print("x", end='', flush=True)

print("")
print(p2.name + " won: " + str(p2.won_games) + " Games")
print(p3.name + " won: " + str(p3.won_games) + " Games")
print("Ties: " + str(total_games - p2.won_games - p3.won_games))
