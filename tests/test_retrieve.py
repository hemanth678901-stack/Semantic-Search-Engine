import pytest
import json
from typing import List
from pathlib import Path
import sys

# Add the project root so pytest can import the local src package.
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))
from src.retrieve_tfidf import TFIDFSearchEngine

def test_file(tmp_path) -> None:
    """This function tests the size of TFIDF search engine."""
    fake_data = [
        {"title": "Doc 1", "text": "This is about Python."},
        {"title": "Doc 2", "text": "This is about Java."},
        {"title": "Doc 3", "text": "This is C++."}
    ]
    # Keep test data isolated in pytest's temporary directory.
    path = tmp_path / "corpus.json"
    with open(path, "w") as file:
        json.dump(fake_data, file, indent=2)
    test1 = TFIDFSearchEngine()
    test1.load_and_fit(path)
    res = test1.search(["About"], 2)
    assert len(res) == 2

def test_search_accuracy(tmp_path) -> None:
    """This function tests the accuracy of TFIDF search engine.."""
    fake_data = [
            {"title": "Doc 1", "text": "This is about Python."},
            {"title": "Doc 2", "text": "This is about Java."},
            {"title": "Doc 3", "text": "This is C++."}
        ]
    # Build a fresh corpus so this test does not depend on test_file().
    path = tmp_path / "corpus.json"
    with open(path, "w") as file:
        json.dump(fake_data, file, indent=2)
    test1 = TFIDFSearchEngine()
    test1.load_and_fit(path)
    res = test1.search(["Python"], 2)
    assert res[0]["title"] == "Doc 1"
