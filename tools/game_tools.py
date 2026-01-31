from lib.tooling import tool
from vector_store.game_vector_store import game_vector_store
@tool
def retrieve_game(query: str, n_results: int = 5):
    """
    Semantic search: Finds the most relevant results in the vector database regarding the game industry.

    Args:
        query: A question about the game industry.

    Returns:
        A list of results. Each element contains:
        - Platform
        - Name
        - YearOfRelease
        - Description
    """

    return game_vector_store.retrieve_games(query, n_results)
