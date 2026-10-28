import math
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]


# Small sample records based on the topics and titles in test_queries.json.
# The data is kept here so this example does not need to load another file.
articles: List[dict] = [
    {
        "title": "Sundar Pichai",
        "text": "Sundar Pichai is the CEO of Google and Alphabet.",
        "topic": "Google leadership",
        "related_questions": ["Who is the CEO of Google?", "Who is the boss of Google?"],
    },
    {
        "title": "Google Gemini",
        "text": "Google Gemini is a family of artificial intelligence models developed by Google.",
        "topic": "Google AI",
        "related_questions": ["What is Google Gemini?"],
    },
    {
        "title": "Alphabet Inc.",
        "text": "Alphabet Inc. is the parent company of Google.",
        "topic": "Technology companies",
        "related_questions": ["What is Alphabet?"],
    },
    {
        "title": "Google",
        "text": "Google is a technology company that provides search and online services.",
        "topic": "Technology companies",
        "related_questions": ["Tell me about Google."],
    },
    {
        "title": "Larry Page",
        "text": "Larry Page is a co-founder of Google.",
        "topic": "Google founders",
        "related_questions": ["Who founded Google?"],
    },
    {
        "title": "Google Drive",
        "text": "Google Drive is a cloud storage service for saving and sharing files.",
        "topic": "Google services",
        "related_questions": ["What is Google Drive?"],
    },
]


class CosineSimilarity:
    """Calculate cosine similarity using simple Python loops."""

    def dot_product(self, first_vector: List[float], second_vector: List[float]) -> float:
        total = 0.0

        for position in range(len(first_vector)):
            total = total + first_vector[position] * second_vector[position]

        return total

    def vector_length(self, vector: List[float]) -> float:
        total = 0.0

        for number in vector:
            total = total + number * number

        return math.sqrt(total)

    def calculate(self, first_vector: List[float], second_vector: List[float]) -> float:
        # Cosine similarity = dot product / (length A * length B).
        top = self.dot_product(first_vector, second_vector)
        bottom = self.vector_length(first_vector) * self.vector_length(second_vector)

        if bottom == 0:
            return 0

        return top / bottom


def find_article(question: str) -> str:
    """Find an article when the question matches one of its sample questions."""
    for article in articles:
        for sample_question in article["related_questions"]:
            if question.lower() == sample_question.lower():
                return article["title"]

    return "No matching article found"


def main() -> None:
    # First, show that a Google CEO question is linked to Sundar Pichai.
    question = "Who is the boss of Google?"
    matching_title = find_article(question)
    print(f"Question: {question}")
    print(f"Matching article: {matching_title}\n")

    # Make one string for each article from its title and text.
    article_texts = []
    for article in articles:
        article_texts.append(article["title"] + ". " + article["text"])

    # Turn those strings into vectors (embeddings) using the project model.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    article_vectors = model.encode(article_texts)

    # Compare the same two articles named in the README.
    sundar_number = 0
    gemini_number = 1
    sundar_vector = [float(number) for number in article_vectors[sundar_number]]
    gemini_vector = [float(number) for number in article_vectors[gemini_number]]

    # Calculate the score with our class, then calculate it with scikit-learn.
    calculator = CosineSimilarity()
    python_score = calculator.calculate(sundar_vector, gemini_vector)
    sundar_array = np.array([sundar_vector])
    gemini_array = np.array([gemini_vector])
    sklearn_score = float(cosine_similarity(sundar_array, gemini_array)[0][0])

    print('Pair: "Sundar Pichai" & "Google Gemini"')
    print(f"Raw Python Math Score: {python_score:.16f}")
    print(f"Production Library Score: {sklearn_score:.16f}")
    print(f"Difference: {abs(python_score - sklearn_score):.3g}")


if __name__ == "__main__":
    main()
