from vector_store.game_vector_store import game_vector_store

def retrieve_game(query: str, n_results: int = 5):
    """
    Semantic search: Finds most relevant results in the vector DB.

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
