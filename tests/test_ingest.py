import pytest
import json
from pathlib import Path

# Build absolute paths to project root and default corpus file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.json"

def test_schema_check() -> None:
    """This function tests the structure of the data."""
    with open(DEFAULT_CORPUS_PATH, "r") as file:
        corpus_data = json.load(file)
    for article in corpus_data:
        assert isinstance(article, dict)
        assert "id" in article
        assert "title" in article
        assert "text" in article

def test_data_quality() -> None:
    """This function tests the quality of the data."""
    with open(DEFAULT_CORPUS_PATH, "r") as file:
        data = json.load(file)
    assert len(data) >= 150
    for article in data:
        assert article["title"] != ""
        assert article["title"] != None
        assert article["text"] != ""
        assert article["text"] != None
