
from Agents.udaplay_agent import UdaPlayAgent

from tools.game_tools import retrieve_game
from vector_store.game_vector_store import game_vector_store
from tools import tool_registry, evaluate_retrieval, game_web_search

game_vector_store.index_games("data/games")

tools = [retrieve_game, evaluate_retrieval, game_web_search]

agent = UdaPlayAgent(tools)

while True:
    q = input("\nAsk UdaPlay: ")
    print("\nAnswer:", agent.run(q))
