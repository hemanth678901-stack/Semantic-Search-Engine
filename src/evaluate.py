import logging
from pathlib import Path
from retrieve_embed import SemanticSearchEngine
from retrieve_tfidf import TFIDFSearchEngine
import json
import time
import numpy as np
from typing import List, Optional

# Build the absolute paths of project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set up the basic logging config.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("System.log")
formatter = logging.Formatter("%(asctime)s - %(message)s - %(levelname)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Evaluate:
    def __init__(self) -> None:
        """This constructor function will do the boot up work when class object was called."""
        self.tfidf = TFIDFSearchEngine()
        self.embed = SemanticSearchEngine()
        logger.info("Loading both the engines.")
        self.tfidf.load_and_fit(PROJECT_ROOT / "data" / "corpus.json")
        self.embed.load_file(PROJECT_ROOT / "data" / "corpus.json")
        logger.info("Loading and fitting the actual data to the engines.")

    def calculate_metric(self, data: List, count: int, gold_set: List) -> Optional[List]:
        """This function will calculate the metrics."""
        current_count = 0
        rank = 0
        for title in data:
            if title["title"] in gold_set[count]["title"]:
                current_count += 1
        for i in range(len(data)):
            if data[i]["title"] in gold_set[count]["title"]:
                rank = i+1
                break
        precision5 = float(current_count / 5)
        recall5 = float(current_count / len(gold_set[count]["title"]))
        reciprocal_rank = float(1 / rank) if rank != 0 else 0
        logger.info("Three mathematical operations has been completed.")
        return [precision5, recall5, reciprocal_rank]
    
    def evaluate(self) -> None:
        """This function will evaluate the result and store the metrics."""
        with open(PROJECT_ROOT / "data" / "test_queries.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        tfidf_metrics = []
        embed_metrics = []
        tfidf_latency = []
        embed_latency = []
        for i in range(len(data)):
            tfidf_start_time = time.perf_counter() 
            tfidf_res = self.tfidf.search([data[i]["query"]], 5)
            tfidf_end_time = time.perf_counter()
            tfidf_latency.append(tfidf_end_time - tfidf_start_time)
            logger.info(f"TFIDF engine has completed the process. The latency for {data[i]} is {tfidf_latency[i]}")
            embed_start_time = time.perf_counter()
            embed_res = self.embed.search(data[i]["query"], 5)
            embed_end_time = time.perf_counter()
            embed_latency.append(embed_end_time - embed_start_time)
            logger.info(f"Embeddings engine has completed the process. The latency for {data[i]} is {embed_latency[i]}")
            logger.info("Calling the metrics function with the result of both engines.")
            tfidf_metrics.append(self.calculate_metric(tfidf_res, i, data))
            embed_metrics.append(self.calculate_metric(embed_res, i, data))
        return self.report_format(tfidf_metrics, embed_metrics, tfidf_latency, embed_latency, len(data))

    def report_format(self, *args) -> None:
        """This function will create and write the final report."""
        # Converting the lists to numpy arrays for fast and accurate calculations.
        logger.info("Calculating the mean of all the metrics.") 
        # Converting the lists to numpy array.
        tfidf_metric_report, embed_metric_report, tfidf_latency_mean, embed_latency_mean = np.array(args[0]), np.array(args[1]), np.array(args[2]), np.array(args[3])
        # Find the sum of the elements in the nested list of metrics.
        tfidf_metric_report, embed_metric_report = np.round(np.sum(args[0], axis=0)), np.round(np.sum(args[1], axis=0))
        # Find the sum of the elements in the nested list of latency.
        tfidf_latency_mean, embed_latency_mean = np.sum(args[2], axis=0), np.sum(args[3], axis=0)
        # Divide the total sum with number of documents that were returned. 
        tfidf_metric_report, embed_metric_report, tfidf_latency_mean, embed_latency_mean = tfidf_metric_report / args[-1], embed_metric_report / args[-1], np.round(tfidf_latency_mean / args[-1], 6), np.round(embed_latency_mean / args[-1], 6)
        # convert the numpy array to standard list.
        tfidf_metric_report, embed_metric_report, tfidf_latency_mean, embed_latency_mean = tfidf_metric_report.tolist(), embed_metric_report.tolist(), tfidf_latency_mean.tolist(), embed_latency_mean.tolist()
        logger.info("Writing the results to the final report.MD")
        try:
            path = Path(PROJECT_ROOT / "results")
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "eval_report.MD", "w") as file:
                file.write(
                    "Metric(Mean)   TFIDF(Sparse)          Dense Embeddings(ChromaDB)\n"
                    "------------   -------------          -------------------------\n"
                    f"Precision@5    {tfidf_metric_report[0]}    {embed_metric_report[0]}\n"
                    "------------   ------------------     --------------------------\n"
                    f"Recall@5       {tfidf_metric_report[1]}    {embed_metric_report[1]}\n"
                    "------------   ------------------     --------------------------\n"
                    f"MRR            {tfidf_metric_report[2]}     {embed_metric_report[2]}\n"
                    "------------   ------------------     --------------------------\n"
                    f"Mean Latency   {tfidf_latency_mean} ms            {embed_latency_mean} ms\n"
                )
                logger.info("Writing the report is completed.")
        except FileNotFoundError as e:
            logger.error("File not found.")

if __name__ == "__main__":
    evaluation = Evaluate()
    evaluation.evaluate()