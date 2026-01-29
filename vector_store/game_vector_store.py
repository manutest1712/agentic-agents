from pathlib import Path
import os
import json
import chromadb
from chromadb.utils import embedding_functions

from config.env import load_env


class GameVectorStore:
    def __init__(
        self,
        chroma_path: str = "./chromadb",
        collection_name: str = "udaplay",
        openai_api_key: str | None = None
    ):
        """
        Initializes ChromaDB client, embedding function, and collection.
        """

        print("Creating game vector store")

        load_env()
        self.chroma_path = Path(chroma_path)

        # Create client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path)
        )

        # Embedding function
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY")
        )

        # Collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    # --------------------------------------------------
    # Indexing
    # --------------------------------------------------
    def index_games(self, data_dir: str):
        """
        Loads all JSON game files from a directory and indexes them.
        """

        for file_name in sorted(os.listdir(data_dir)):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)

            content = (
                f"[{game['Platform']}] "
                f"{game['Name']} ({game['YearOfRelease']}) - "
                f"{game['Description']}"
            )

            doc_id = os.path.splitext(file_name)[0]

            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[game]
            )

            print(f"Added file to collection: {file_path}")

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    def retrieve_games(self, query: str, n_results: int = 5):
        """
        Performs semantic search on the game collection.
        """

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        games = []
        for metadata in results["metadatas"][0]:
            games.append({
                "Platform": metadata.get("Platform"),
                "Name": metadata.get("Name"),
                "YearOfRelease": metadata.get("YearOfRelease"),
                "Description": metadata.get("Description")
            })

        return games


game_vector_store = GameVectorStore()