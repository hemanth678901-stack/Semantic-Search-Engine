import logging
import requests
import time
from pathlib import Path
import json
from  typing import List, Optional

# Setting up the basic logging config.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("System.log", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(message)s - %(levelname)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Build absolute paths for the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A Custome Exception class for APILimit.
class APILIMIT(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

def search(topic: str) -> List[str]:
    """This function will search the titles of various articles."""
    # Wikipedia Action API is used for searching articles.
    search_url = "https://en.wikipedia.org/w/api.php" 
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": 500,
        "format": "json"
    }
    header = {
        "User-Agent": "ingest",
        "Accept": "application/json"
    }
    search_response = requests.get(search_url, headers=header, params=params, timeout=5)
    if search_response.status_code != 200:
        return []
    search_data = search_response.json()["query"]["search"]
    all_titles = []
    for title in search_data:
        all_titles.append(title.get("title"))
    return all_titles

def api_request(article: str) -> Optional[dict]:
    """This function will fetch the summary of the pages."""
    # Wikipedia Rest API is used for the fetching the summary of the pages.
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}"
    headers = {
        "User-Agent": "ingest/v1.0",
        "Accept": "application/json"
    }
    while True:
        try:
            response = requests.get(url, timeout=5, headers=headers)
            if response.status_code == 200:
                data = response.json()
                clean_data = {
                    "id": str(data.get("pageid")),
                    "title": data["title"],
                    "text": data["extract"]
                }
                return clean_data
            elif response.status_code == 429 or response.status_code == 509:
                raise APILIMIT("Server is busy with too many requests.")
        except APILIMIT as e:
            logger.error("API: Too Many Requests")
            time.sleep(5)

# A few topics for finding the articles.
TOPICS = [
    "Sundar Pichai",
    "Machine learning algorithms",
    "Database management systems",
    "Computer networking protocols",
    "Operating system kernels",
    "Cryptography algorithms",
    "Web development frameworks"
]

if __name__ == "__main__":
    path = PROJECT_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    final_path = path / "corpus.json"
    if final_path.exists():
        with open(final_path, "r") as file:
            all_articles = json.load(file)
        exist_ids = {article["id"] for article in all_articles}
    else:
        # Empty list and set for all articles and existing ids.
        all_articles = []
        exist_ids = set()

    for title in TOPICS:
        titles = search(title)
        for article in titles:
            clean_data = api_request(article)
            if clean_data is not None and clean_data["id"] not in exist_ids:
                all_articles.append(clean_data)
                exist_ids.add(clean_data["id"])
                logger.info(f"Articles for {article} is fetched.")
            else:
                logger.info(f"Articles for {article} is not fetched.")
    with open(final_path, "w", encoding="utf-8") as file: 
        json.dump(all_articles, file, indent=2)