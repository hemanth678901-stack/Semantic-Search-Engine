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
| **Precision@5** | 0.176923076923077 | 0.22307692307692317
| **Recall@5** | 0.12637362637362634 | 0.15934065934065927
| **MRR** | 0.6634615384615384 | 0.9070512820512822
| **Mean Latency** | 0.001786 ms | 0.01141 ms

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

I compared my calculation with `sklearn.metrics.pairwise.cosine_similarity` using sentence embeddings from `all-MiniLM-L6-v2`. The demo creates those embeddings from six short sample article records written directly in `cosine_similarity_demo.py`; they are illustrative examples based on the project's query topics, not articles loaded from the corpus. Both calculations produce the same result apart from tiny floating-point rounding differences.

**Verification Example:**
*   **Pair:** "Sundar Pichai" & "Google Gemini"
*   **Raw Python Math Score:** `0.4472105200087608`
*   **Production Library Score:** `0.4472105200087604`


## ⚠️ System Limitations

**Domain Out-of-Distribution:** Queries on topics outside the ingested corpus cannot return valid facts.

**Vocabulary Mismatch(TF-IDF):** Sparse matching fails entirely when queries use synonyms or paraphrases without exact token overlap.

**Inference Latency(Dense Embeddings):** Passing tect through transformer attention layers is computationally heavier than sparse matrix loopups, requiring GPU acceleration or quantization at large production scale.
