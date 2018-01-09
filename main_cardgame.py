""" Let the Game begin """
# Arguments when starting from console: main_cardgame.py [total_games] [Number of X for progress]
import sys
from cardgame.random_ai import RandomAIPlayer
from cardgame.plain_agent import RLAgent
from cardgame.real_player import RealPlayer
from cardgame.game import Game

#p1 = RealPlayer("Niko")

p2 = RandomAIPlayer ("John")
p3 = RLAgent ("Stefan", hidden_layers=20, learning_rate=1e-2, update_frequency=10)

total_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
progress_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 10

g = Game()
g.init_game(p2, p3)
for i in range(total_games):
    g.start_new_match()
    if i % (total_games / progress_splits) == 0:
        print("x", end='', flush=True)

g.finalize_game()
print("")
print(p2.name + " won: " + str(p2.won_games) + " Games")
print(p3.name + " won: " + str(p3.won_games) + " Games")
print("Ties: " + str(total_games - p2.won_games - p3.won_games))
