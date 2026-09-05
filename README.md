# Semantic Search Engine: Lexical(TF-IDF) vs Dense Embeddings

A comparative search and retrieval system demonstrating quantitative and qualitative differences between sparse lexical search (TF-IDF) and dense semantic retrieval (Sentence Transformers + ChromaDB).

----

## 🏗️ System Architecture

The pipeline consists of four modular components.

1. **`src/ingest.py:` (Data Pipeline):** Programmatically queries the Wikipedia Search & Rest APIs across multiple technical domains, sanitizes text extracts, handles rate limit gracefully with backoff retries, and saves a structured corpus (`data/corpus.json`).

2. **`src/retrieve_tfidf.py` (Sparse Baseline):** Fits a `TFIDFVectorizer` over the entire corpus vocabulary and computes cosine similarity against sparse document matrices for exac keyword matching.

3. **`src/retrieve_embed.py` (Dense Semantic Engine):** Encodes text passages into a 384-dimensional dense vectors using `all-MiniLM-L6-v2` and persists them into a local `ChromaDB` collection configured with cosine distance and HNSW indexing.

4. **`src/evaluate.py` (Evaluation Harness):** Benchmarks both engines against a hand-labelled ground-truth gold set (`data/test_queries.json`) to compute comparative ranking and latency metrics.

---

## 📊 Benchmark & Evaluation Results

Benchmarked on a gold-set test suite across exact-match, semantic-intent, and synonym heavy queries:

| Metric (Mean) | TF-IDF (Sparse) | Dense Embeddings (ChromaDB) |
| :--- | :--- | :--- |
| **Precision@5** | 0.19230769230769232 | 0.23076923076923078
| **Recall@5** | 0.11538461538461539 | 0.15384615384615385
| **MRR** | 0.6538461538461539 | 0.9230769230769231
| **Mean Latency** | 0.001232 ms | 0.007576 ms

---

## 🚀 Setup & Installation

```bash
# 1. Create the virtual environment.
python -m venv .venv
```

```bash
# 2. Activate the virtual environment(Windows Powershell)
.\venv\Scripts\Activate.ps1
```

```bash
# 3. Install all the dependencies
pip install -r requirements.txt
```

```bash
# Run retrieval via CLI
python main.py --engine both --query {user-query} "Who is the CEO of Alphabet?" --top_k 5
```

```bash
# Run automated test suite
python -m pytest
```


### Vector Mathematics Verification

I verified that the core algorithms powering this search engine match fundamental linear algebra. I implemented the cosine similarity calculation from scratch in pure python using custom dot product and vector magnitude functions.

I tested my raw math against the production library (`sklearn.metrics.pairwise.cosine_similarity`) using real embeddings from this project's corpus. The results matched exactly ( accounting for standard floating-point precision differences at the 15th decimal place ).

**Verification Example:**
*   **Pair:** "Sundar Pichai" & "Google Gemini"
*   **Raw Python Math Score:** `0.4709807120973254`
*   **Production Library Score:** `0.4709807120973256`


## ⚠️ System Limitations

**Domain Out-of-Distribution:** Queries on topics outside the ingested corpus cannot return valid facts.

**Vocabulary Mismatch(TF-IDF):** Sparse matching fails entirely when queries use synonyms or paraphrases without exact token overlap.

**Inference Latency(Dense Embeddings):** Passing tect through transformer attention layers is computationally heavier than sparse matrix loopups, requiring GPU acceleration or quantization at large production scale.