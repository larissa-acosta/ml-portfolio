"""Gradio demo (Hugging Face Space): ask a question about LGPD, Marco Civil or CDC.

Retrieval runs locally (BM25 + multilingual-e5 hybrid). Generation uses the Hugging Face
Inference API, so the Space needs a secret named HF_TOKEN. Set LLM_ID to any chat model
available through the Inference API.
"""
import os
import sys
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient

sys.path.insert(0, str(Path(__file__).parent / "src"))
from generate import build_messages  # noqa: E402
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, load_chunks  # noqa: E402

LLM_ID = os.environ.get("LLM_ID", "Qwen/Qwen2.5-7B-Instruct")
K = 5

chunks = load_chunks("data/chunks.jsonl")
retriever = HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, "intfloat/multilingual-e5-base")])
client = InferenceClient(token=os.environ.get("HF_TOKEN"))


def ask(question: str):
    if not question.strip():
        return "Digite uma pergunta.", ""
    ctx = [chunks[i] for i, _ in retriever.search(question, K)]
    try:
        resp = client.chat_completion(messages=build_messages(question, ctx), model=LLM_ID,
                                      max_tokens=500, temperature=0.1)
        answer = resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        answer = f"Erro ao chamar o modelo ({type(e).__name__}). Verifique HF_TOKEN e LLM_ID."
    sources = "\n\n".join(f"**{c['law']} {c['article']}**: {c['text'][:400]}…" for c in ctx)
    return answer, sources


demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Pergunta", placeholder="Em quantos dias posso desistir de uma compra online?"),
    outputs=[gr.Markdown(label="Resposta"), gr.Markdown(label="Trechos recuperados")],
    title="RAG jurídico em português: LGPD, Marco Civil e CDC",
    description="Projeto acadêmico. Não substitui aconselhamento jurídico. Respostas com citação dos artigos.",
)

if __name__ == "__main__":
    demo.launch()
