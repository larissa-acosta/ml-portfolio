"""Compare BM25, dense and hybrid retrieval with hit@k and MRR.

Usage:
    python src/eval_retrieval.py --questions data/eval_questions.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, load_chunks

MAX_RANK = 10


def unique_articles(results: list[tuple[int, float]], chunks: list[dict]) -> list[str]:
    seen, out = set(), []
    for idx, _ in results:
        a = chunks[idx]["article_id"]
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def evaluate(search_fn, questions: list[dict], chunks: list[dict], ks=(1, 3, 5, 10)):
    """search_fn(query, k) -> [(chunk_idx, score)]. Returns (metrics dict, per-question rows)."""
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    rows = []
    for q in questions:
        ranked = unique_articles(search_fn(q["question"], 50), chunks)[:MAX_RANK]
        gold = set(q["gold"])
        rank = next((i for i, a in enumerate(ranked, start=1) if a in gold), None)
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        rr_sum += 1.0 / rank if rank else 0.0
        rows.append({"id": q["id"], "question": q["question"], "gold": ",".join(q["gold"]),
                     "rank": rank, "top3": ",".join(ranked[:3])})
    n = len(questions)
    metrics = {f"hit@{k}": hits[k] / n for k in ks}
    metrics["MRR"] = rr_sum / n
    return metrics, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--questions", default="data/eval_questions.jsonl")
    ap.add_argument("--dense-models", nargs="*", default=["intfloat/multilingual-e5-base", "BAAI/bge-m3"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]
    print(f"{len(chunks)} chunks, {len(questions)} perguntas")

    bm25 = BM25Retriever(chunks)
    retrievers = [bm25]
    dense = [DenseRetriever(chunks, m) for m in args.dense_models]
    retrievers += dense
    retrievers += [HybridRetriever([bm25, d]) for d in dense]

    table, failures = [], []
    for r in retrievers:
        metrics, rows = evaluate(r.search, questions, chunks)
        table.append({"retriever": r.name, **{k: round(v, 3) for k, v in metrics.items()}})
        failures += [{"retriever": r.name, **row} for row in rows if row["rank"] is None or row["rank"] > 5]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(table)
    df.to_csv(out / "retrieval_results.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "retrieval_failures.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSalvo em {out}/retrieval_results.csv e retrieval_failures.csv")


if __name__ == "__main__":
    main()
