import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np

# Importing the packages
import chromadb
from sentence_transformers import SentenceTransformer

# Setup the standard logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("System.log")
formatter = logging.Formatter("%(asctime)s - %(message)s - %(levelname)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Build absolute paths to project root and default corpus file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.json"

class SemanticSearchEngine:
    def __init__(self) -> None:
        # Loading the model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model was loaded successfully.")
        self.client = chromadb.PersistentClient(path=PROJECT_ROOT / "data" / "chromadb") # Creating the folder for database.
        logger.info("Folder for the database is created.")
        # Creating a document for the embeddings.
        self.collection = self.client.get_or_create_collection(
            name = "wikipedia-corpus",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Wikipedia corpus embeddings was created.")

    def load_file(self,path: Path) -> None:
        """This function will load data and add the embeddings to the database."""
        # Load embeddings only when the collection is empty.
        if self.collection.count() == 0:
            # Try for exception handling
            try:
                with open(path, "r") as file:
                    data = json.load(file)
                # Refining the data before embeddings.
                ids = [article["id"] for article in data]
                metadatas_list = [{"title": t["title"]} for t in data]
                text = [article["text"] for article in data]
                embeddings = self.model.encode(text, batch_size=32, show_progress_bar=True)
                # Embeddings the data to the Wikipedia-Corpus document.
                self.collection.add(
                    ids = ids,
                    documents=text,
                    embeddings=embeddings,
                    metadatas=metadatas_list #type: ignore
                )
                logger.info("Embeddings of articles was successfully added to the database.")
            # Exception is used to avoid the crash of program when file is not found.
            except FileNotFoundError as e:  
                logger.error("File was not found.")

    def search(self, user_query: str, top_k: int) -> List[Dict]:
        """This function will search the articles relevant to user query."""
        # Converting the user query to dense vectors.
        user_embed = self.model.encode(user_query, batch_size=32).tolist()
        # Searching for the relevant articles in database. The Math is done by the database(chromadb) internally.
        res = self.collection.query(
            query_embeddings=[user_embed],
            n_results=top_k
        )
        logger.info("Data search is completed.")
        title = [meta["title"] for meta in res["metadatas"][0]]#type: ignore
        text = list(res["documents"][0]) #type: ignore
        distance = list(res["distances"][0]) #type: ignore
        # Combining the all the result into a combined data in organized format.
        final_res = [{"title": t, "text": txt, "distance": dis} for t, txt, dis in zip(title, text, distance)]
        return final_res

if __name__ == "__main__":
    Engine = SemanticSearchEngine()
    Engine.load_file(DEFAULT_CORPUS_PATH)
    print(Engine.search("Who is the Boss of Alphabet?", 2))