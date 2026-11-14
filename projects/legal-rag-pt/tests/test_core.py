"""Light tests (no network, no GPU, no embedding models). Run: python tests/test_core.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_retrieval import evaluate  # noqa: E402
from generate import extract_citations  # noqa: E402
from ingest import html_to_lines, parse_articles  # noqa: E402
from retrieval import BM25Retriever, rrf  # noqa: E402

SAMPLE_HTML = """
<html><body>
<p>LEI Nº 99, DE 1 DE JANEIRO DE 2020</p>
<p>O PRESIDENTE DA REPÚBLICA decreta:</p>
<p>CAPÍTULO I</p>
<p>DISPOSIÇÕES PRELIMINARES</p>
<p>Art. 1o  Esta Lei dispõe sobre o tratamento de dados pessoais. (Redação dada pela Lei nº 1/2021)</p>
<p>Parágrafo único.  As normas gerais são de interesse nacional.</p>
<p>Art. 2º  A disciplina da proteção de dados tem como fundamentos:</p>
<p>I - o respeito à privacidade;</p>
<p>II - a autodeterminação informativa.</p>
<p>Art. 3º (Revogado pela Lei nº 5/2021)</p>
<p>Art. 10.  O controlador poderá tratar dados para finalidades legítimas.</p>
<p>Art. 18-A.  Fica criado o direito de exemplo para testes automatizados.</p>
<p>Brasília, 1 de janeiro de 2020; 199º da Independência.</p>
<p>FULANO DE TAL</p>
</body></html>
""".encode("utf-8")


def test_parse_articles():
    chunks = parse_articles(html_to_lines(SAMPLE_HTML), "TST", "Lei de Teste")
    ids = [c["article_id"] for c in chunks]
    assert ids == ["TST-1", "TST-2", "TST-10", "TST-18A"], ids
    assert "Redação dada" not in chunks[0]["text"]
    assert "Parágrafo único" in chunks[0]["text"]
    assert "autodeterminação" in chunks[1]["text"]
    assert "FULANO" not in chunks[-1]["text"] and "Brasília" not in chunks[-1]["text"]
    assert "DISPOSIÇÕES" not in " ".join(c["text"] for c in chunks)


def test_long_article_split():
    long_lines = ["Art. 1º Texto inicial."] + [f"Inciso {i} - " + "palavra " * 40 for i in range(20)]
    chunks = parse_articles(long_lines, "TST", "Lei de Teste")
    assert len(chunks) > 1
    assert {c["article_id"] for c in chunks} == {"TST-1"}
    assert [c["part"] for c in chunks] == list(range(len(chunks)))


def test_rrf_and_bm25():
    assert rrf([[1, 2, 3], [3, 1, 4]])[0][0] == 1  # 1 is high in both lists
    chunks = [
        {"chunk_id": "A-1#0", "article_id": "A-1", "text": "O consumidor pode desistir da compra em sete dias."},
        {"chunk_id": "A-2#0", "article_id": "A-2", "text": "O titular dos dados tem direito de acesso."},
        {"chunk_id": "A-3#0", "article_id": "A-3", "text": "Provedores guardam registros de conexão por um ano."},
    ]
    bm25 = BM25Retriever(chunks)
    assert bm25.search("desistir da compra", 1)[0][0] == 0
    assert bm25.search("acesso do titular", 1)[0][0] == 1


def test_evaluate_metrics():
    chunks = [{"article_id": f"X-{i}"} for i in range(5)]
    questions = [
        {"id": "a", "question": "q1", "gold": ["X-0"]},   # rank 1
        {"id": "b", "question": "q2", "gold": ["X-2"]},   # rank 3
        {"id": "c", "question": "q3", "gold": ["X-9"]},   # miss
    ]

    def fake_search(q, k):
        return [(i, 1.0) for i in range(5)]

    m, rows = evaluate(fake_search, questions, chunks, ks=(1, 3))
    assert m["hit@1"] == 1 / 3 and m["hit@3"] == 2 / 3
    assert abs(m["MRR"] - (1 + 1 / 3) / 3) < 1e-9
    assert rows[2]["rank"] is None


def test_citations():
    text = "Pode desistir em 7 dias [CDC Art. 49] e ver seus dados [LGPD Art. 18] ou [CDC Art. 18-A]."
    assert extract_citations(text) == ["CDC-49", "LGPD-18", "CDC-18A"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
