import argparse
from src.retrieve_tfidf import TFIDFSearchEngine
from src.retrieve_embed import SemanticSearchEngine
from pathlib import Path

# Build absolute paths to project root and default corpus file.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.json"

# Create the top-level parser
parse = argparse.ArgumentParser()

# Create a argument for search engine
parse.add_argument("--engine", type=str, choices=["embed", "tfidf", "both"], required=True)

# Create a argument for user query
parse.add_argument("--query", type=str, required=True)
parse.add_argument("--top_k",type=int, default=5)


if __name__ == "__main__":
    # Parse the inputs
    args = parse.parse_args()
    # Core logic of the program.
    if args.engine == "embed":
        engine = SemanticSearchEngine()
        engine.load_file(DEFAULT_CORPUS_PATH)
        semantic_search_res = engine.search(args.query, args.top_k)
        print("Semantic Search output: ")
        print("Title: Distance")
        # Create a loop to print in the correct format.
        for i in range(len(semantic_search_res)):
            print(f"{i+1}. {semantic_search_res[i]['title']} - {semantic_search_res[i]['distance']}")

    elif args.engine == "tfidf":
        engine = TFIDFSearchEngine()
        engine.load_and_fit(DEFAULT_CORPUS_PATH)
        tfidf_res = engine.search([args.query], args.top_k)
        print("TF-IDF search output: ")
        print("Title: Score")
        # Create a loop to print in the correct format.
        for i in range(len(tfidf_res)):
            print(f"{i+1}. {tfidf_res[i]['title']} - {tfidf_res[i]['score']}")

    else:
        tfidf_engine = TFIDFSearchEngine()
        tfidf_engine.load_and_fit(DEFAULT_CORPUS_PATH)
        tfidf_res = tfidf_engine.search([args.query], args.top_k)
        semantic_search_engine = SemanticSearchEngine()
        semantic_search_engine.load_file(DEFAULT_CORPUS_PATH)
        semantic_search_res = semantic_search_engine.search(args.query, args.top_k)
        print("TF-IDF Search")
        print("Title: Score")
        # Create a loop to print in the correct format.
        for i in range(args.top_k):
            print(f"{i+1}. {tfidf_res[i]['title']} - {tfidf_res[i]['score']}")
        print("------------------------------------------------")
        print("Semantic Search")
        print("Title: Distance")
        for i in range(len(semantic_search_res)):
            print(f"{i+1}. {semantic_search_res[i]['title']}: {semantic_search_res[i]['distance']}")
