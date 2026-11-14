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

Article-level retrieval on 57 hand-written questions (an answer counts as correct if any gold article is retrieved):

| Retriever | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.596 | 0.719 | 0.807 | 0.912 | 0.692 |
| Dense (e5-base) | 0.526 | 0.825 | 0.877 | 0.930 | 0.679 |
| **Dense (bge-m3)** | **0.860** | **0.930** | **0.982** | **1.000** | **0.906** |
| Hybrid (BM25 + e5-base) | 0.579 | 0.842 | 0.895 | 0.982 | 0.722 |
| Hybrid (BM25 + bge-m3) | 0.667 | 0.860 | 0.930 | 1.000 | 0.777 |

**Takeaways**

- Dense bge-m3 alone is the best retriever on every metric. It puts a correct article first for 86% of questions and in the top 10 for all of them.
- The hybrids were worse than bge-m3 alone. With equal-weight Reciprocal Rank Fusion, the weaker BM25 and e5 rankings pulled the result down. I did not tune the fusion weights, so this says "untuned RRF did not help here", not that hybrid retrieval cannot help.
- The demo therefore uses bge-m3 only.

**Error analysis** (from `results/retrieval_failures.csv`, which lists every question whose gold article is not in the top 5)

- BM25 misses paraphrases with little word overlap, for example "Quando o tratamento de dados pessoais deve ser encerrado?" (gold LGPD Art. 15, which talks about "término") or "Em que casos os dados pessoais podem ser conservados depois do fim do tratamento?" (LGPD Art. 16). Neither gold article appears in its top 10.
- Definition-style questions with two gold articles are hard for BM25 and for both hybrids, for example "Quem é considerado consumidor e quem é considerado fornecedor?" (CDC Art. 2 and 3): none of them has a gold article in the top 5, while bge-m3 alone does.
- The only bge-m3 miss in the top 5 is "Quais penalidades se aplicam a quem descumpre as regras de guarda de registros?" (gold MCI Art. 12): it ranks the gold article 8th, behind neighboring articles of the same law and two unrelated CDC articles (top 3: MCI-13, CDC-73, CDC-72).
- Generation has not been evaluated yet (`src/eval_generation.py`).

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

`data/eval_questions.jsonl` has 57 questions (LGPD, Marco Civil da Internet and CDC), each with the expected article (`gold`). The gold articles were checked against the downloaded law text (the article exists and contains the expected provision), but the legal reading behind each pair is the project authors' and has not been reviewed by a lawyer. A question counts as a hit when **any** of its gold articles is retrieved; questions that need several articles at once, or span two laws, are not covered yet.

## Limitations

- Evaluation set is small (57 questions) and written by the project authors.
- Groundedness is measured by an embedding-similarity proxy plus citation checks; it is not a human judgment of legal correctness.
- Laws change; the ingestion downloads the current compiled versions.
- No handling of questions spanning several laws beyond what retrieval returns.

## Demo

🤗 Space (retrieval only, bge-m3): https://huggingface.co/spaces/larissaac/legal-rag-pt

The Space shows the retrieval step. Cited generation is in this repository (`src/generate.py`, `src/eval_generation.py`) and needs a model: a local one through `transformers`, or an inference API.

## My contribution

Personal project, built end to end: I defined the problem and the evaluation protocol (article-level retrieval metrics and citation checks), and implemented the ingestion, the BM25 / dense / hybrid retrievers, the cited-answer generation, and the evaluation scripts. Built with AI assistance (Claude), which I reviewed and am validating against the law text.
