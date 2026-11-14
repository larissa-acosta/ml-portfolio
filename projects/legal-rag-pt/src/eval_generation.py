"""Evaluate generated answers: citation accuracy and groundedness.

Metrics (all automatic proxies, plus a CSV for manual review):
  - citation_hit: the gold article is cited in the answer
  - citation_valid: fraction of cited articles that were actually in the retrieved context
  - grounded_sim: cosine similarity between the answer and the retrieved context (embedding model)
  - abstained: the model said it could not answer

Usage:
    python src/eval_generation.py --questions data/eval_questions.jsonl --k 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate import NO_ANSWER, answer, extract_citations, load_generator
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, load_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--questions", default="data/eval_questions.jsonl")
    ap.add_argument("--dense-model", default="BAAI/bge-m3")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]

    dense = DenseRetriever(chunks, args.dense_model)
    retriever = HybridRetriever([BM25Retriever(chunks), dense])
    generator = load_generator(args.llm)

    rows = []
    for q in questions:
        text, ctx = answer(q["question"], retriever, chunks, generator, k=args.k)
        cited = extract_citations(text)
        ctx_articles = {c["article_id"] for c in ctx}
        ctx_text = "\n".join(c["text"] for c in ctx)

        prefix_q, prefix_p = ("query: ", "passage: ") if dense.is_e5 else ("", "")
        a_emb = dense.model.encode([prefix_q + text], normalize_embeddings=True)[0]
        c_emb = dense.model.encode([prefix_p + ctx_text], normalize_embeddings=True)[0]

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "gold": ",".join(q["gold"]),
            "answer": text,
            "cited": ",".join(cited),
            "citation_hit": any(g in cited for g in q["gold"]),
            "citation_valid": (sum(c in ctx_articles for c in cited) / len(cited)) if cited else np.nan,
            "grounded_sim": float(a_emb @ c_emb),
            "abstained": NO_ANSWER.lower() in text.lower(),
            "gold_in_context": any(g in ctx_articles for g in q["gold"]),
            "manual_correct (preencher)": "",
        })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "generation_results.csv", index=False)

    print(f"Perguntas: {len(df)}")
    print(f"Gabarito no contexto recuperado: {df['gold_in_context'].mean():.2%}")
    print(f"Citou o artigo correto:          {df['citation_hit'].mean():.2%}")
    print(f"Citações válidas (no contexto):  {df['citation_valid'].mean():.2%}")
    print(f"Similaridade resposta-contexto:  {df['grounded_sim'].mean():.3f}")
    print(f"Abstenções:                      {df['abstained'].mean():.2%}")
    print(f"\nRevise manualmente: {out}/generation_results.csv")


if __name__ == "__main__":
    main()
