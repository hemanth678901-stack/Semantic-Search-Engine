# Import the packages
import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List


# Setup of logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("System.log")
formatter = logging.Formatter("%(asctime)s - %(message)s - %(levelname)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Build absolute paths to project root and default corpus file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.json"

class TFIDFSearchEngine:
    def __init__(self) -> None:
        # The vectorizer learns the corpus vocabulary during load_and_fit().
        self.vectorizer = TfidfVectorizer()
        self.master_document = None

    def load_and_fit(self, path: Path) -> None:
        """This function will load the data and compute TF-IDF vectors."""
         # Load document text and preserve titles for the returned search results.
        try:
            with open(path, "r") as file:
                data = json.load(file)
            text_list = [txt["text"] for txt in data]
            self.title_list = [txt["title"] for txt in data]
            self.master_document = self.vectorizer.fit_transform(text_list)
            logger.info(f"Data is successfully processed. The length of the master document is {len(self.title_list)}")
        except FileNotFoundError as e:
            logger.error(f"File Not Found.")

    def search(self, user_query: List[str], top_k: int) -> List[dict]:
        """This function will transform the user query and compute the cosine similarity scores for finding similarity."""
        # Transform the query with the same vocabulary fitted on the corpus.
        user_matrix = self.vectorizer.transform(user_query)
        scores = cosine_similarity(user_matrix, self.master_document)
        sorted_index = np.argsort(scores[0])[::-1][:top_k]
        res = [
            {
                "title": self.title_list[index],
                "score": float(scores[0][index])
            }
            for index in sorted_index
        ]
        return res


if __name__ == "__main__":
    s = TFIDFSearchEngine()
    s.load_and_fit(DEFAULT_CORPUS_PATH)
    print(s.search(["Sundar Pichai Machine Learning Google"], 3))