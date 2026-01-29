from tools.game_tools import retrieve_game
from vector_store.game_vector_store import game_vector_store

game_vector_store.index_games("data/games")

print("retrieving games")
retrieve_game("ss");