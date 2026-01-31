from tools.evaluation_tools import evaluate_retrieval
from tools.game_tools import retrieve_game
from tools.tool_registry import ToolRegistry
from tools.game_web_search import game_web_search

tool_registry = ToolRegistry()

tool_registry.register(retrieve_game)
tool_registry.register(evaluate_retrieval)
tool_registry.register(game_web_search)
