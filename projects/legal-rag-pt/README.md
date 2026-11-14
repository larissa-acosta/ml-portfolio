# Legal RAG in Portuguese: LGPD, Marco Civil da Internet and CDC

A retrieval-augmented generation system that answers questions about Brazilian data-protection, internet and consumer law, **citing the article it relied on**, with a systematic comparison of retrieval strategies.

> ⚠️ Academic project. Not legal advice.

## Why this project

Legal text has a natural structure (articles), answers can be checked against a ground-truth article, and a wrong citation is easy to detect. That makes it a good setting to **evaluate** a RAG system instead of only demoing it.

## Pipeline

1. **Ingestion** (`src/ingest.py`): downloads the three laws from the official Planalto website and chunks them **by article** (long articles are split on line boundaries and keep the same `article_id`). Revoked articles and editorial annotations are removed.
2. **Retrieval** (`src/retrieval.py`): BM25 (lexical), dense retrieval with multilingual embeddings (`multilingual-e5-base`, `bge-m3`) and a **hybrid** retriever using Reciprocal Rank Fusion.
3. **Generation** (`src/generate.py`): an open instruction-tuned LLM answers only from the retrieved articles, must cite them as `[LEI Art. N]` and must abstain when the context is insufficient.
4. **Evaluation**
   - `src/eval_retrieval.py`: hit@k and MRR at article level.
   - `src/eval_generation.py`: citation accuracy, citation validity (no invented articles), answer-context similarity (groundedness proxy) and a CSV for manual review.

## Results

`TODO: fill in after running the evaluation.`

| Retriever | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| BM25 | | | | | |
| Dense (e5-base) | | | | | |
| Dense (bge-m3) | | | | | |
| Hybrid (BM25 + e5-base) | | | | | |
| Hybrid (BM25 + bge-m3) | | | | | |

Generation (`TODO: model, k`): gold article in context `TODO`% · cites correct article `TODO`% · valid citations `TODO`% · abstentions `TODO`%.

**Error analysis:** `TODO: 2-3 typical failure patterns, e.g. questions that mix two laws, or paraphrases with no lexical overlap.`

## Run it

```bash
pip install -r requirements.txt
python src/ingest.py                      # builds data/chunks.jsonl
python src/eval_retrieval.py              # retrieval comparison -> results/
python src/eval_generation.py --k 5       # needs a GPU for the default 3B model
python tests/test_core.py                 # light unit tests, no network or GPU
```

Or open `run_colab.ipynb` in Google Colab (upload the repository as a zip).

## Evaluation set

`data/eval_questions.jsonl` has 20 starter questions with the expected article (`gold`). **The gold articles are a first draft and must be checked against the law text**; the goal is to grow this set to 50-80 questions, including some that need more than one article.

## Limitations

- Evaluation set is small and written by the project authors.
- Groundedness is measured by an embedding-similarity proxy plus citation checks; it is not a human judgment of legal correctness.
- Laws change; the ingestion downloads the current compiled versions.
- No handling of questions spanning several laws beyond what retrieval returns.

## Demo

🤗 Space: `TODO` (`app.py`; set the `HF_TOKEN` secret and optionally `LLM_ID`)

## My contribution

Personal project, built end to end: I defined the problem and the evaluation protocol (article-level retrieval metrics and citation checks), and implemented the ingestion, the BM25 / dense / hybrid retrievers, the cited-answer generation, and the evaluation scripts. Built with AI assistance (Claude), which I reviewed and am validating against the law text.
