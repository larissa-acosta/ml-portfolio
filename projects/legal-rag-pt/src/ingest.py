"""Download Brazilian laws from the Planalto website and split them by article.

Usage:
    python src/ingest.py            # writes data/chunks.jsonl (and caches raw HTML in data/raw/)
"""
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

LAWS = {
    "LGPD": {
        "name": "Lei Geral de Proteção de Dados (Lei 13.709/2018)",
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
    },
    "MCI": {
        "name": "Marco Civil da Internet (Lei 12.965/2014)",
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm",
    },
    "CDC": {
        "name": "Código de Defesa do Consumidor (Lei 8.078/1990)",
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
    },
}

MAX_CHARS = 1800  # articles longer than this are split into parts (same article_id)

ARTICLE_RE = re.compile(r"^Art\.?\s*(\d+)\s*(?:[º°]|o\b)?\s*(?:-\s*([A-Z])\b)?\s*[.\-–]?\s*(.*)$")
ANNOTATION_RE = re.compile(
    r"\((?:Redação dada|Incluíd[oa]|Vide|Revogad[oa]|Vigência|Regulamento|Produção de efeito|Promulgação partes vetadas)[^)]*\)",
    flags=re.IGNORECASE,
)
HEADING_RE = re.compile(r"^(PARTE|LIVRO|TÍTULO|TITULO|CAPÍTULO|CAPITULO|SEÇÃO|SECAO|SUBSEÇÃO|Seção|Subseção)\b")
END_RE = re.compile(r"^Bras[íi]lia,")


def fetch_html(law_id: str, url: str, raw_dir: Path) -> bytes:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"{law_id}.html"
    if cache.exists():
        data = cache.read_bytes()
        if len(data) < 5000 or b"Acesso Negado" in data:
            cache.unlink()  # a previously cached error page; download again
        else:
            return data
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (legal-rag-pt research project)"})
    resp.raise_for_status()
    if len(resp.content) < 5000 or b"Acesso Negado" in resp.content:
        raise SystemExit(
            f"The Planalto website blocked the download of {law_id} (common on Colab / cloud IPs).\n"
            f"Open {url} in your browser, save the page as data/raw/{law_id}.html "
            f"(Ctrl+S, 'Web page, HTML only') and run this script again: it uses the saved file."
        )
    cache.write_bytes(resp.content)
    return resp.content


def decode_html(html: bytes) -> str:
    """Planalto pages are not always UTF-8 (usually Windows-1252 / ISO-8859-1).
    Guessing the encoding wrongly garbles accents, so decode explicitly."""
    try:
        return html.decode("utf-8")
    except UnicodeDecodeError:
        return html.decode("cp1252", errors="replace")


def html_to_lines(html: bytes) -> list[str]:
    soup = BeautifulSoup(decode_html(html), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Planalto keeps superseded / revoked wording struck through: drop it, keep only the text in force
    for tag in soup.find_all(["s", "strike", "del"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"line-through", re.I)):
        tag.decompose()
    text = soup.get_text("\n")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
        if line:
            lines.append(line)
    return lines


def clean(text: str) -> str:
    text = ANNOTATION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_articles(lines: list[str], law_id: str, law_name: str) -> list[dict]:
    """Split a list of text lines into article chunks."""
    articles = []  # list of (number, suffix, [lines])
    current = None
    for line in lines:
        if END_RE.match(line) and current is not None:
            break
        m = ARTICLE_RE.match(line)
        if m:
            current = (m.group(1), m.group(2) or "", [line])
            articles.append(current)
            continue
        if current is None:
            continue  # preamble
        if HEADING_RE.match(line) or (line.isupper() and len(line) < 100):
            continue  # structural headings
        current[2].append(line)

    chunks = []
    seen = {}
    for number, suffix, art_lines in articles:
        cleaned = [clean(l) for l in art_lines]
        text = "\n".join(l for l in cleaned if l)
        body = re.sub(r"^Art\.?\s*\d+\s*(?:[º°]|o\b)?\s*(?:-\s*[A-Z]\b)?\s*[.\-–]?\s*", "", text).strip()
        if len(body) < 10:
            continue  # revoked or empty article
        article_id = f"{law_id}-{number}{suffix}"
        label = f"Art. {number}{'-' + suffix if suffix else ''}"
        # split long articles on line boundaries
        parts, buf = [], ""
        for l in text.split("\n"):
            if buf and len(buf) + len(l) + 1 > MAX_CHARS:
                parts.append(buf)
                buf = l
            else:
                buf = f"{buf}\n{l}" if buf else l
        if buf:
            parts.append(buf)
        n = seen.get(article_id, 0)
        seen[article_id] = n + 1
        if n:  # duplicated article numbers (rare): keep them distinguishable
            article_id = f"{article_id}.{n}"
        for p, part_text in enumerate(parts):
            chunks.append({
                "chunk_id": f"{article_id}#{p}",
                "article_id": article_id,
                "law": law_id,
                "law_name": law_name,
                "article": label,
                "part": p,
                "text": part_text,
            })
    return chunks


def main(out_path: str = "data/chunks.jsonl") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out.parent / "raw"
    all_chunks = []
    for law_id, meta in LAWS.items():
        html = fetch_html(law_id, meta["url"], raw_dir)
        chunks = parse_articles(html_to_lines(html), law_id, meta["name"])
        n_art = len({c["article_id"] for c in chunks})
        print(f"{law_id}: {n_art} artigos, {len(chunks)} chunks")
        all_chunks.extend(chunks)
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Total: {len(all_chunks)} chunks -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
