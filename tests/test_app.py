import pytest
from src.app import EmptyContentError, write_file
from pathlib import Path

def test_empty_content_error(tmp_path):
    with pytest.raises(EmptyContentError):
        write_file(tmp_path , "Nothing", "")
