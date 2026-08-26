from pathlib import Path
import pytest
import json
from src.retrieve_embed import SemanticSearchEngine

# Fake document is created for the test purpose.
fake_document = {
    "id": "25173473",
    "title": "Sundar Pichai",
    "text": "Pichai Sundararajan, better known as Sundar Pichai, is an Indian, 2013American business executive who has been the CEO of Google since 2015 and the CEO of its parent company Alphabet Inc. since 2019."
}

def test_retrieve(tmp_path: Path) -> None:
    """This function tests the size and accuracy of the semantic search engine."""
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as file:
        json.dump([fake_document], file, indent=2)
    s1 = SemanticSearchEngine()
    s1.load_file(file_path)
    assert len(s1.search("Who is the CEO of parent company of google?", 1)) == 1
    assert s1.search("Who is the CEO of parent company of google?", 1)[0]["title"] == "Sundar Pichai"
