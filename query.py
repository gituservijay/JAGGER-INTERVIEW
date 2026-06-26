#!/usr/bin/env python3
"""
query.py — Answer a natural-language question using BM25 retrieval + Gemini.

Usage:
    python query.py --index ./index.pkl -q "What is the procurement threshold?"
    python query.py --index ./index.pkl -q "..." --top-k 5

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
# Google Gemini
# ---------------------------------------------------------------------------
try:
    import google.generativeai as genai
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai", "-q"])
    import google.generativeai as genai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_index(index_path: str) -> dict:
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve(index_data: dict, question: str, top_k: int) -> list:
    """BM25 retrieval — returns top_k chunks sorted by score."""
    bm25 = index_data["bm25"]
    chunks = index_data["chunks"]
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    seen_docs = {}
    results = []
    for score, chunk in scored:
        doc_id = chunk["doc_id"]
        if doc_id not in seen_docs:
            seen_docs[doc_id] = score
            results.append((score, chunk))
        if len(results) >= top_k:
            break

    return results


def build_prompt(question: str, retrieved: list) -> str:
    context_parts = []
    for i, (score, chunk) in enumerate(retrieved, 1):
        context_parts.append(f"[Document {i}: {chunk['doc_id']}]\n{chunk['text']}")
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


def answer_question(
    index_path: str,
    question: str,
    top_k: int = 5,
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

    if verbose:
        print(f"\nPrompt size: ~{len(prompt)//4} tokens (estimated)")

    # Setup Gemini
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: No API key found. Set GEMINI_API_KEY environment variable or pass --api-key")
        sys.exit(1)

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Call LLM
    t2 = time.time()
    response = model.generate_content(prompt)
    t_llm = time.time() - t2

    answer = response.text.strip()

    # Token usage
    try:
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
    except Exception:
        input_tokens = len(prompt) // 4
        output_tokens = len(answer) // 4

    total_tokens = input_tokens + output_tokens

    # Gemini 1.5 Flash pricing: ~$0.075/M input, $0.30/M output
    cost_usd = (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    print(f"\n{answer}")
    print(f"\n{'─'*60}")
    print(f"Sources retrieved (top {top_k}):")
    for i, (score, chunk) in enumerate(retrieved, 1):
        print(f"  {i}. {chunk['doc_id']}  (BM25 score: {score:.3f})")
    print(f"\nCost metrics:")
    print(f"  Model:          gemini-1.5-flash")
    print(f"  Input tokens:   {input_tokens}")
    print(f"  Output tokens:  {output_tokens}")
    print(f"  Total tokens:   {total_tokens}")
    print(f"  Est. cost:      ${cost_usd:.6f}")
    print(f"  Retrieval time: {t_retrieve*1000:.1f}ms (BM25, local)")
    print(f"  LLM time:       {t_llm:.2f}s")
    print(f"{'─'*60}")

    return {
        "question": question,
        "answer": answer,
        "sources": [c["doc_id"] for _, c in retrieved],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the BM25 index using Gemini.")
    parser.add_argument("--index", default="./index.pkl", help="Path to index file")
    parser.add_argument("--question", "-q", required=True, help="Natural language question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default 5)")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show retrieval details")
    args = parser.parse_args()

    answer_question(
        index_path=args.index,
        question=args.question,
        top_k=args.top_k,
        api_key=args.api_key,
        verbose=args.verbose,
    )
 

    
