# Document Q&A Beyond RAG — BM25 Approach

## Approach

This system replaces embedding-based vector retrieval with **BM25** (Best Match 25), a classic probabilistic keyword ranking algorithm. BM25 is the backbone of search engines like Elasticsearch and Lucene.

**Pipeline:**
```
Corpus → Chunker → BM25 Index (disk)
                        ↓
Question → BM25 Scorer → Top-K Chunks → Claude (Haiku) → Answer + Citations
```

### Why BM25 instead of RAG?

| | RAG (embeddings) | This system (BM25) |
|---|---|---|
| Core retrieval | Dense vector search | Keyword/TF-IDF ranking |
| Index cost | Embed every chunk (LLM API calls) | Free (pure math) |
| Per-query retrieval cost | Vector DB query | Free (local, <5ms) |
| Per-query LLM cost | top-k chunks → LLM | top-k chunks → LLM (same) |
| Cross-doc questions | Weak | Strong (global scoring) |
| Exact term matching | Weak | Strong |
| Semantic similarity | Strong | Weak |

**Cost discipline:** BM25 retrieval is entirely local (no API calls). Only the LLM step costs money, and it receives the same ~5 chunks a RAG system would. Per-query cost is essentially identical to RAG.

---

## Setup

```bash
pip install rank_bm25 anthropic pdfplumber
export ANTHROPIC_API_KEY=your_key_here
```

---

## Step 1 — Index the corpus

```bash
python index.py --corpus ./corpus --output ./index.pkl
```

Options:
- `--corpus`     Path to directory containing your documents (`.txt`, `.pdf`, `.md`)
- `--output`     Where to save the index (default: `./index.pkl`)
- `--chunk-size` Words per chunk (default: 800)
- `--overlap`    Word overlap between consecutive chunks (default: 100)

This is a **one-time cost**. The index is ~10–50 MB for hundreds of documents and loads in under 1 second.

---

## Step 2 — Ask a question

```bash
python query.py --index ./index.pkl --question "What is the procurement threshold for defence contracts?"
```

Options:
- `--question` / `-q`  Your natural language question (required)
- `--top-k`            Chunks to retrieve (default: 5)
- `--model`            Anthropic model (default: `claude-haiku-4-5-20251001` for lowest cost)
- `--verbose` / `-v`   Show BM25 scores and retrieval details
- `--api-key`          API key (or set `ANTHROPIC_API_KEY` env var)

### Example output

```
============================================================
QUESTION: What is the procurement threshold for defence contracts?
============================================================

Based on Document 3 (procurement-policy-2023.txt), the procurement threshold
for defence contracts is £100,000 for goods and services...

──────────────────────────────────────────────────────────
Sources retrieved (top 5):
  1. procurement-policy-2023.txt  (BM25 score: 12.341)
  2. defence-framework-v2.txt     (BM25 score: 8.102)
  ...

Cost metrics:
  Model:         claude-haiku-4-5-20251001
  Input tokens:  2847
  Output tokens: 134
  Total tokens:  2981
  Est. cost:     $0.002813
  Retrieval time: 3.2ms (BM25, local)
  LLM time:      1.43s
```

---

## Handling different question types

| Question type | How BM25 handles it |
|---|---|
| Specific fact in one doc | BM25 scores exact terms highly → correct doc rises to top |
| Needle-in-a-haystack | BM25 global ranking finds the one doc with the matching terms |
| Multi-document synthesis | top-k set covers multiple relevant docs; LLM synthesises |
| Whole-corpus / thematic | BM25 returns a diverse high-scoring set; works reasonably well |
| Unanswerable | LLM instructed to say "UNANSWERABLE" if context is insufficient |

---

## Cost comparison vs RAG baseline

Assuming **5 chunks × 800 words ≈ 4,000 words ≈ 5,000 tokens** sent to the LLM per query:

| Metric | RAG | This system |
|---|---|---|
| Retrieval API cost | ~$0.0001 (vector DB query) | $0.00 (local BM25) |
| LLM input tokens | ~5,000 | ~5,000 |
| LLM output tokens | ~200 | ~200 |
| Total cost (Haiku) | ~$0.005 | ~$0.004 |
| Index freshness cost | Re-embed changed docs | Re-index (free) |

BM25 is **slightly cheaper** than RAG per query because retrieval itself is free.

---

## Tradeoffs & what I'd improve

**Where BM25 degrades vs RAG:**
- Synonyms / paraphrasing: "cost" vs "price" vs "expenditure" — BM25 misses if exact terms don't match. *Fix: add synonym expansion or a light query-rewriting step.*
- Semantic questions: "What documents discuss fiscal responsibility?" — RAG handles this better.
- Corpus growth: re-indexing is fast (seconds) but must be re-run manually.

**What I'd add with more time:**
1. Query expansion using a cheap LLM call to generate synonyms before BM25 scoring
2. A re-ranker (e.g. cross-encoder) as a second pass over top-20 BM25 results
3. Metadata filters (date, document type) to narrow the candidate set
4. Streaming output for better UX

---

## Evaluation metrics

- **Answer quality**: Manual spot-check + exact-match on factual questions with known answers
- **Cost per query**: Logged automatically (tokens + USD)
- **Unanswerable accuracy**: Does the system say "UNANSWERABLE" when it should?
- **Latency**: BM25 retrieval (<10ms) + LLM (~1–3s)
