"""RAG answer generation with an open instruction-tuned model."""
import re

SYSTEM_PROMPT = (
    "Você é um assistente jurídico que responde SOMENTE com base nos trechos de lei fornecidos. "
    "Responda em português, de forma objetiva. Depois de cada afirmação, cite o artigo no formato "
    "[LEI Art. N], por exemplo [LGPD Art. 18] ou [CDC Art. 49]. "
    "Se os trechos não forem suficientes para responder, diga exatamente: "
    "'Não encontrei essa informação nos trechos fornecidos.' Não invente artigos."
)

NO_ANSWER = "Não encontrei essa informação nos trechos fornecidos."
# tolerant to small format deviations of the model: "[LEI MCI Art. 13]", "CDC Art. 18" (no brackets)
CITATION_RE = re.compile(r"\[?(?:LEI\s+)?(LGPD|MCI|CDC)\s+Art\.?\s*(\d+)(?:\s*-\s*([A-Z]))?")


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['law']} {c['article']}] ({c['law_name']})\n{c['text']}" for c in chunks)


def build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    user = f"Trechos da legislação:\n\n{format_context(context_chunks)}\n\nPergunta: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(answer: str) -> list[str]:
    """Return cited article ids in the same format as the chunks (e.g. 'LGPD-18', 'CDC-18A')."""
    return [f"{law}-{num}{suffix or ''}" for law, num, suffix in CITATION_RE.findall(answer)]


def load_generator(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    from transformers import pipeline

    return pipeline("text-generation", model=model_name, device_map="auto", torch_dtype="auto")


def answer(question: str, retriever, chunks: list[dict], generator, k: int = 5, max_new_tokens: int = 700):
    """Retrieve top-k chunks and generate an answer. Returns (answer_text, retrieved_chunks)."""
    idxs = [i for i, _ in retriever.search(question, k)]
    context = [chunks[i] for i in idxs]
    out = generator(build_messages(question, context), max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip(), context
