#!/usr/bin/env python3


import argparse
import os
import pickle
import re
import sys
from pathlib import Path

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ---------------------------------------------------------------------------
# BM25 — pure Python, no embedding, no vector DB
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank_bm25 ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank_bm25", "-q"])
    from rank_bm25 import BM25Okapi



def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def read_file(path: Path) -> str:
    """Return plain text content of a file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not PDF_SUPPORT:
            print(f"  [skip] PDF support not available (pip install pdfplumber): {path.name}")
            return ""
        text_parts = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
        except Exception as e:
            print(f"  [warn] Could not read PDF {path.name}: {e}")
        return "\n".join(text_parts)
    else:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [warn] Could not read {path.name}: {e}")
            return ""


def chunk_text(text: str, doc_id: str, chunk_size: int = 800, overlap: int = 100):
    """
    Split text into overlapping word-level chunks.
    Returns list of dicts: {doc_id, chunk_index, text}
    """
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk_text,
        })
        if end == len(words):
            break
        start += chunk_size - overlap
        idx += 1
    return chunks




def build_index(corpus_dir: str, output_path: str, chunk_size: int = 800, overlap: int = 100):
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"ERROR: corpus directory not found: {corpus_dir}")
        sys.exit(1)

    # Gather files
    extensions = {".txt", ".md", ".pdf"}
    files = [f for f in corpus_path.rglob("*") if f.suffix.lower() in extensions and f.is_file()]
    print(f"Found {len(files)} documents in {corpus_dir}")

    all_chunks = []
    for i, fpath in enumerate(sorted(files)):
        print(f"  [{i+1}/{len(files)}] {fpath.name}")
        text = read_file(fpath)
        if not text.strip():
            continue
        doc_id = str(fpath.relative_to(corpus_path))
        chunks = chunk_text(text, doc_id, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Build BM25
    print("Building BM25 index ...")
    tokenized = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    # Save
    index_data = {
        "bm25": bm25,
        "chunks": all_chunks,
        "num_docs": len(files),
        "num_chunks": len(all_chunks),
    }
    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"\n✓ Index saved to {output_path}")
    print(f"  Documents: {len(files)}")
    print(f"  Chunks:    {len(all_chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 index over a document corpus.")
    parser.add_argument("--corpus", default="./corpus", help="Path to corpus directory")
    parser.add_argument("--output", default="./index.pkl", help="Output index file path")
    parser.add_argument("--chunk-size", type=int, default=800, help="Words per chunk (default 800)")
    parser.add_argument("--overlap", type=int, default=100, help="Word overlap between chunks (default 100)")
    args = parser.parse_args()

    build_index(args.corpus, args.output, args.chunk_size, args.overlap)
