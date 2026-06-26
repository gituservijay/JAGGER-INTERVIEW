#!/usr/bin/env python3
"""
query.py — Answer a natural-language question using BM25 retrieval + Claude.

Usage:
    python query.py --index ./index.pkl --question "What is the procurement threshold?"
    python query.py --index ./index.pkl --question "..." --top-k 5 --model claude-haiku-4-5-20251001

Cost model:
    BM25 retrieval is free (local).
    Only top-k chunks (~800 words each) are sent to the LLM.
    Default top-k=5 → ~4000 words context per query ≈ comparable to a RAG query.
"""

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank_bm25", "-q"])
    from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
    import anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_index(index_path: str) -> dict:
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve(index_data: dict, question: str, top_k: int) -> list[dict]:
    """BM25 retrieval — returns top_k chunks sorted by score."""
    bm25: BM25Okapi = index_data["bm25"]
    chunks: list[dict] = index_data["chunks"]
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)

    # Pair chunks with scores and sort
    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )
    # Deduplicate by doc_id — take best chunk per doc first, then fill up
    seen_docs = {}
    results = []
    for score, chunk in scored:
        doc_id = chunk["doc_id"]
        if doc_id not in seen_docs:
            seen_docs[doc_id] = score
            results.append((score, chunk))
        if len(results) >= top_k:
            break

    return results  # list of (score, chunk)


def build_prompt(question: str, retrieved: list[tuple]) -> str:
    context_parts = []
    for i, (score, chunk) in enumerate(retrieved, 1):
        context_parts.append(
            f"[Document {i}: {chunk['doc_id']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a precise document Q&A assistant. Answer the question using ONLY the provided document excerpts.

Rules:
- If the answer is found, state it clearly and cite the source document(s) by name.
- If the documents do not contain enough information to answer, say exactly: "UNANSWERABLE: The corpus does not contain sufficient information to answer this question."
- Do not hallucinate or use outside knowledge.
- Keep your answer concise and factual.

---
DOCUMENTS:
{context}

---
QUESTION: {question}

ANSWER:"""


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def answer_question(
    index_path: str,
    question: str,
    top_k: int = 5,
    model: str = "claude-haiku-4-5-20251001",
    api_key: str = None,
    verbose: bool = False,
):
    # Load index
    t0 = time.time()
    index_data = load_index(index_path)
    t_load = time.time() - t0
    if verbose:
        print(f"Index loaded in {t_load:.2f}s  ({index_data['num_chunks']} chunks, {index_data['num_docs']} docs)")

    # Retrieve
    t1 = time.time()
    retrieved = retrieve(index_data, question, top_k=top_k)
    t_retrieve = time.time() - t1
    if verbose:
        print(f"\nBM25 retrieval in {t_retrieve*1000:.1f}ms")
        for score, chunk in retrieved:
            print(f"  score={score:.3f}  doc={chunk['doc_id']}  chunk={chunk['chunk_index']}")

    # Build prompt
    prompt = build_prompt(question, retrieved)
    input_tokens_est = estimate_tokens(prompt)

    if verbose:
        print(f"\nPrompt size: ~{input_tokens_est} tokens (estimated)")

    # Call LLM
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    t2 = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    t_llm = time.time() - t2

    answer = message.content[0].text.strip()
    input_tokens_actual = message.usage.input_tokens
    output_tokens_actual = message.usage.output_tokens
    total_tokens = input_tokens_actual + output_tokens_actual

    # Cost estimate (Haiku pricing as of 2025)
    # claude-haiku-4-5: $0.80/M input, $4/M output
    cost_usd = (input_tokens_actual * 0.80 + output_tokens_actual * 4.0) / 1_000_000

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    print(f"\n{answer}")
    print(f"\n{'─'*60}")
    print(f"Sources retrieved (top {top_k}):")
    for i, (score, chunk) in enumerate(retrieved, 1):
        print(f"  {i}. {chunk['doc_id']}  (BM25 score: {score:.3f})")
    print(f"\nCost metrics:")
    print(f"  Model:         {model}")
    print(f"  Input tokens:  {input_tokens_actual}")
    print(f"  Output tokens: {output_tokens_actual}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Est. cost:     ${cost_usd:.6f}")
    print(f"  Retrieval time: {t_retrieve*1000:.1f}ms (BM25, local)")
    print(f"  LLM time:      {t_llm:.2f}s")
    print(f"{'─'*60}")

    return {
        "question": question,
        "answer": answer,
        "sources": [c["doc_id"] for _, c in retrieved],
        "input_tokens": input_tokens_actual,
        "output_tokens": output_tokens_actual,
        "cost_usd": cost_usd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the BM25 document index with LLM answering.")
    parser.add_argument("--index", default="./index.pkl", help="Path to index file")
    parser.add_argument("--question", "-q", required=True, help="Natural language question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default 5)")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model to use (default: claude-haiku-4-5-20251001 for lowest cost)",
    )
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show retrieval details")
    args = parser.parse_args()

    answer_question(
        index_path=args.index,
        question=args.question,
        top_k=args.top_k,
        model=args.model,
        api_key=args.api_key,
        verbose=args.verbose,
    )
