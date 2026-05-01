#!/usr/bin/env python3
"""60-question retrieval benchmark across 5 chunking strategies.

Questions are categorized as:
  - factual: single-fact lookup
  - multi_part: questions that need information from multiple sections
  - structure_aware: questions about specific algorithms, properties, formal definitions

Embeds queries one at a time to avoid llama-cpp batch sequence-id bug.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import faiss

from src.embedder import SentenceTransformer


QUESTIONS = [
    # FACTUAL (20)
    {"q": "What problem does concurrency control solve in a database system?",
     "category": "factual",
     "keywords": ["concurrent", "consistency", "transactions", "isolation"],
     "section_fragment": "Transaction Isolation"},
    {"q": "What is the two-phase locking protocol?",
     "category": "factual",
     "keywords": ["growing phase", "shrinking phase", "two-phase", "locking"],
     "section_fragment": "Two-Phase Locking"},
    {"q": "What is timestamp-based concurrency control?",
     "category": "factual",
     "keywords": ["timestamp", "ordering", "read", "write"],
     "section_fragment": "Timestamp"},
    {"q": "What is snapshot isolation?",
     "category": "factual",
     "keywords": ["snapshot", "version", "read", "isolation"],
     "section_fragment": "Snapshot"},
    {"q": "What is write-ahead logging?",
     "category": "factual",
     "keywords": ["log", "before", "disk", "write-ahead"],
     "section_fragment": "Recovery"},
    {"q": "What is a B+ tree index used for?",
     "category": "factual",
     "keywords": ["B+", "tree", "leaf", "search"],
     "section_fragment": "B+"},
    {"q": "What is a hash join?",
     "category": "factual",
     "keywords": ["hash", "join", "partition", "build"],
     "section_fragment": "Hash"},
    {"q": "What is a foreign key constraint?",
     "category": "factual",
     "keywords": ["foreign key", "reference", "constraint"],
     "section_fragment": "Foreign"},
    {"q": "What is a primary key?",
     "category": "factual",
     "keywords": ["primary key", "unique", "identifier"],
     "section_fragment": "Primary"},
    {"q": "What is normalization in database design?",
     "category": "factual",
     "keywords": ["normal form", "dependency", "decomposition"],
     "section_fragment": "Normal"},
    {"q": "What is a serializable schedule?",
     "category": "factual",
     "keywords": ["serializable", "schedule", "equivalent", "serial"],
     "section_fragment": "Serializ"},
    {"q": "What is a deadlock in a database system?",
     "category": "factual",
     "keywords": ["deadlock", "wait", "transactions", "circular"],
     "section_fragment": "Deadlock"},
    {"q": "What is a relational schema?",
     "category": "factual",
     "keywords": ["schema", "relation", "attributes", "structure"],
     "section_fragment": "Relational"},
    {"q": "What is the SQL SELECT statement used for?",
     "category": "factual",
     "keywords": ["select", "query", "tuple", "from"],
     "section_fragment": "SQL"},
    {"q": "What is referential integrity?",
     "category": "factual",
     "keywords": ["referential", "integrity", "foreign key", "constraint"],
     "section_fragment": "Referential"},
    {"q": "What is a view in SQL?",
     "category": "factual",
     "keywords": ["view", "virtual", "table", "query"],
     "section_fragment": "View"},
    {"q": "What is a cursor in database programming?",
     "category": "factual",
     "keywords": ["cursor", "row", "fetch", "iterator"],
     "section_fragment": "Cursor"},
    {"q": "What is a stored procedure?",
     "category": "factual",
     "keywords": ["stored", "procedure", "function", "executable"],
     "section_fragment": "Stored"},
    {"q": "What is a checkpoint in database recovery?",
     "category": "factual",
     "keywords": ["checkpoint", "log", "consistent", "state"],
     "section_fragment": "Checkpoint"},
    {"q": "What is a transaction in a database system?",
     "category": "factual",
     "keywords": ["transaction", "atomic", "operations", "unit"],
     "section_fragment": "Transaction"},

    # MULTI-PART (20)
    {"q": "What is the difference between strict two-phase locking and basic two-phase locking?",
     "category": "multi_part",
     "keywords": ["strict", "exclusive", "commits", "cascading"],
     "section_fragment": "Two-Phase Locking"},
    {"q": "How are deadlocks detected in database systems?",
     "category": "multi_part",
     "keywords": ["deadlock", "wait-for graph", "cycle", "detection"],
     "section_fragment": "Deadlock"},
    {"q": "How does ARIES recovery work?",
     "category": "multi_part",
     "keywords": ["analysis", "redo", "undo", "log"],
     "section_fragment": "ARIES"},
    {"q": "How does the buffer manager handle page replacement?",
     "category": "multi_part",
     "keywords": ["buffer", "page", "replacement", "LRU"],
     "section_fragment": "Buffer"},
    {"q": "What are the ACID properties of a transaction?",
     "category": "multi_part",
     "keywords": ["atomicity", "consistency", "isolation", "durability"],
     "section_fragment": "Transaction"},
    {"q": "How does query optimization choose between alternative plans?",
     "category": "multi_part",
     "keywords": ["plan", "cost", "optimizer", "estimate"],
     "section_fragment": "Query Optimization"},
    {"q": "What is the difference between dense and sparse indexes?",
     "category": "multi_part",
     "keywords": ["dense", "sparse", "index", "entries"],
     "section_fragment": "Index"},
    {"q": "How do hash indexes differ from tree indexes?",
     "category": "multi_part",
     "keywords": ["hash", "tree", "search", "range"],
     "section_fragment": "Hash"},
    {"q": "What is the difference between optimistic and pessimistic concurrency control?",
     "category": "multi_part",
     "keywords": ["optimistic", "pessimistic", "validation", "locking"],
     "section_fragment": "Concurrency"},
    {"q": "How is conflict serializability tested using a precedence graph?",
     "category": "multi_part",
     "keywords": ["conflict", "serializability", "precedence", "graph", "cycle"],
     "section_fragment": "Serializ"},
    {"q": "What is the difference between physical and logical logging?",
     "category": "multi_part",
     "keywords": ["physical", "logical", "log", "operations"],
     "section_fragment": "Log"},
    {"q": "How does a query optimizer estimate the cost of a join operation?",
     "category": "multi_part",
     "keywords": ["cost", "estimate", "join", "selectivity"],
     "section_fragment": "Cost"},
    {"q": "How are functional dependencies used in normalization?",
     "category": "multi_part",
     "keywords": ["functional", "dependency", "normal form", "decomposition"],
     "section_fragment": "Functional"},
    {"q": "What is the difference between volatile, non-volatile, and stable storage?",
     "category": "multi_part",
     "keywords": ["volatile", "non-volatile", "stable", "storage"],
     "section_fragment": "Storage"},
    {"q": "How does multiversion concurrency control reduce contention?",
     "category": "multi_part",
     "keywords": ["multiversion", "version", "snapshot", "concurrent"],
     "section_fragment": "Multivers"},
    {"q": "How are nested loop joins different from sort-merge joins?",
     "category": "multi_part",
     "keywords": ["nested loop", "sort-merge", "join", "comparison"],
     "section_fragment": "Join"},
    {"q": "How is recovery handled after a system crash?",
     "category": "multi_part",
     "keywords": ["recovery", "crash", "redo", "undo", "log"],
     "section_fragment": "Recovery"},
    {"q": "What is the difference between equi-join and natural join?",
     "category": "multi_part",
     "keywords": ["equi-join", "natural", "common", "attribute"],
     "section_fragment": "Join"},
    {"q": "How does the database handle long-running queries with respect to locks?",
     "category": "multi_part",
     "keywords": ["lock", "wait", "long-running", "blocking"],
     "section_fragment": "Lock"},
    {"q": "What is the relationship between transactions and the log?",
     "category": "multi_part",
     "keywords": ["transaction", "log", "record", "commit"],
     "section_fragment": "Log"},

    # STRUCTURE-AWARE (20)
    {"q": "What is query optimization?",
     "category": "structure_aware",
     "keywords": ["query", "plan", "cost", "optimizer"],
     "section_fragment": "Query Optimization"},
    {"q": "What is the structure of a B+ tree leaf node?",
     "category": "structure_aware",
     "keywords": ["leaf", "node", "pointer", "key"],
     "section_fragment": "B+"},
    {"q": "What are the steps of the ARIES recovery algorithm?",
     "category": "structure_aware",
     "keywords": ["analysis", "redo", "undo", "phases"],
     "section_fragment": "ARIES"},
    {"q": "What does the third normal form require?",
     "category": "structure_aware",
     "keywords": ["3NF", "third normal form", "transitive", "dependency"],
     "section_fragment": "Normal Form"},
    {"q": "What is the structure of a transaction log record?",
     "category": "structure_aware",
     "keywords": ["log record", "LSN", "previous", "transaction id"],
     "section_fragment": "Log"},
    {"q": "What is Boyce-Codd Normal Form?",
     "category": "structure_aware",
     "keywords": ["BCNF", "Boyce-Codd", "determinant", "key"],
     "section_fragment": "Boyce"},
    {"q": "What is the structure of an SQL JOIN clause?",
     "category": "structure_aware",
     "keywords": ["JOIN", "ON", "USING", "condition"],
     "section_fragment": "Join"},
    {"q": "How are GROUP BY and HAVING clauses structured in SQL?",
     "category": "structure_aware",
     "keywords": ["GROUP BY", "HAVING", "aggregation"],
     "section_fragment": "Aggregate"},
    {"q": "What is the structure of a relational algebra projection operation?",
     "category": "structure_aware",
     "keywords": ["projection", "pi", "attributes", "relation"],
     "section_fragment": "Relational Algebra"},
    {"q": "What is the structure of a relational algebra selection operation?",
     "category": "structure_aware",
     "keywords": ["selection", "sigma", "predicate", "tuple"],
     "section_fragment": "Relational Algebra"},
    {"q": "What is the structure of a hash table for indexing?",
     "category": "structure_aware",
     "keywords": ["hash", "bucket", "function", "collision"],
     "section_fragment": "Hash"},
    {"q": "How is a heap file organized?",
     "category": "structure_aware",
     "keywords": ["heap", "file", "page", "record"],
     "section_fragment": "Heap"},
    {"q": "What is the structure of a database page?",
     "category": "structure_aware",
     "keywords": ["page", "header", "slot", "record"],
     "section_fragment": "Page"},
    {"q": "What is the structure of a wait-for graph?",
     "category": "structure_aware",
     "keywords": ["wait-for", "graph", "edge", "node", "transaction"],
     "section_fragment": "Deadlock"},
    {"q": "How is the precedence graph constructed for testing serializability?",
     "category": "structure_aware",
     "keywords": ["precedence", "graph", "edge", "conflict", "transaction"],
     "section_fragment": "Serializ"},
    {"q": "What is the structure of a logical log record in ARIES?",
     "category": "structure_aware",
     "keywords": ["LSN", "TransID", "PrevLSN", "PageID", "type"],
     "section_fragment": "ARIES"},
    {"q": "What is the structure of an entity-relationship diagram?",
     "category": "structure_aware",
     "keywords": ["entity", "relationship", "attribute", "diagram"],
     "section_fragment": "Entity"},
    {"q": "How is a foreign key constraint specified in SQL?",
     "category": "structure_aware",
     "keywords": ["FOREIGN KEY", "REFERENCES", "constraint", "table"],
     "section_fragment": "Foreign"},
    {"q": "What is the structure of an SQL CREATE TABLE statement?",
     "category": "structure_aware",
     "keywords": ["CREATE TABLE", "column", "type", "constraint"],
     "section_fragment": "Create"},
    {"q": "What is the structure of a query execution plan tree?",
     "category": "structure_aware",
     "keywords": ["plan", "tree", "operator", "node", "execution"],
     "section_fragment": "Query"},
]


STRATEGIES = [
    ("recursive_sections", "sections", "textbook_index"),
    ("sliding_window", "sliding_window", "textbook_sliding_window"),
    ("sentence_boundary", "sentence_boundary", "textbook_sentence_boundary"),
    ("paragraph", "paragraph", "textbook_paragraph"),
    ("adaptive", "adaptive", "textbook_adaptive"),
]


def load_index(folder, prefix):
    base = Path("index") / folder
    fp = base / f"{prefix}.faiss"
    if not fp.exists():
        return None
    index = faiss.read_index(str(fp))
    with open(base / f"{prefix}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(base / f"{prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {"index": index, "chunks": chunks, "meta": meta}


def keyword_recall(text, keywords):
    if not keywords:
        return 0.0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower) / len(keywords)


def section_hit(meta_entries, fragment):
    frag = fragment.lower()
    return any(frag in m.get("section_path", "").lower() for m in meta_entries)


def embed_queries(embedder, queries):
    embs = []
    for i, q in enumerate(queries):
        e = embedder.encode([q], batch_size=1, show_progress_bar=False,
                            convert_to_numpy=True)
        norm = np.linalg.norm(e[0])
        if norm < 0.1:
            e = embedder.encode([q], batch_size=1, show_progress_bar=False,
                                convert_to_numpy=True)
            norm = np.linalg.norm(e[0])
        embs.append(e[0])
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries)}]")
    return np.array(embs, dtype=np.float32)


def benchmark_strategy(strategy, folder, prefix, query_embs, top_k=10):
    bundle = load_index(folder, prefix)
    if bundle is None:
        return {"strategy": strategy, "status": "MISSING"}

    index = bundle["index"]
    chunks = bundle["chunks"]
    meta = bundle["meta"]

    sizes = [len(c) for c in chunks]
    avg_size = float(np.mean(sizes))
    median_size = float(np.median(sizes))

    t0 = time.time()
    distances, indices = index.search(query_embs, top_k)
    search_time = time.time() - t0

    per_q = []
    for qi, qd in enumerate(QUESTIONS):
        retrieved_chunks = [chunks[idx] for idx in indices[qi]]
        retrieved_meta = [meta[idx] for idx in indices[qi]]
        per_q.append({
            "q": qd["q"],
            "category": qd["category"],
            "top1_recall": keyword_recall(retrieved_chunks[0], qd["keywords"]),
            "top5_recall": keyword_recall(" ".join(retrieved_chunks[:5]), qd["keywords"]),
            "top1_section_hit": section_hit(retrieved_meta[:1], qd["section_fragment"]),
            "top5_section_hit": section_hit(retrieved_meta[:5], qd["section_fragment"]),
        })

    overall = {
        "strategy": strategy,
        "status": "OK",
        "n_chunks": len(chunks),
        "avg_chunk_size": avg_size,
        "median_chunk_size": median_size,
        "search_time_total": search_time,
        "search_time_per_query": search_time / len(QUESTIONS),
        "mean_top1_recall": float(np.mean([r["top1_recall"] for r in per_q])),
        "mean_top5_recall": float(np.mean([r["top5_recall"] for r in per_q])),
        "top1_section_hit_rate": float(np.mean([r["top1_section_hit"] for r in per_q])),
        "top5_section_hit_rate": float(np.mean([r["top5_section_hit"] for r in per_q])),
    }

    # Per-category breakdown
    for cat in ["factual", "multi_part", "structure_aware"]:
        cat_results = [r for r in per_q if r["category"] == cat]
        if cat_results:
            overall[f"{cat}_top1_recall"] = float(np.mean([r["top1_recall"] for r in cat_results]))
            overall[f"{cat}_top5_recall"] = float(np.mean([r["top5_recall"] for r in cat_results]))
            overall[f"{cat}_top1_section_hit"] = float(np.mean([r["top1_section_hit"] for r in cat_results]))
            overall[f"{cat}_top5_section_hit"] = float(np.mean([r["top5_section_hit"] for r in cat_results]))

    overall["per_question"] = per_q
    return overall


def main():
    embed_model = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    print(f"Loading {embed_model}\n")
    embedder = SentenceTransformer(embed_model)

    queries = [q["q"] for q in QUESTIONS]
    print(f"Embedding {len(queries)} queries one-at-a-time")
    t0 = time.time()
    query_embs = embed_queries(embedder, queries)
    print(f"Done in {time.time()-t0:.1f}s. Shape: {query_embs.shape}\n")

    results = []
    for strategy, folder, prefix in STRATEGIES:
        print(f">>> {strategy}")
        try:
            r = benchmark_strategy(strategy, folder, prefix, query_embs)
            results.append(r)
            if r["status"] == "OK":
                print(f"  Chunks: {r['n_chunks']} | Avg: {r['avg_chunk_size']:.0f} chars")
                print(f"  Overall: T1 kw={r['mean_top1_recall']:.3f}  T5 kw={r['mean_top5_recall']:.3f}")
                print(f"  By category (T5 kw):")
                print(f"    factual:         {r.get('factual_top5_recall', 0):.3f}")
                print(f"    multi_part:      {r.get('multi_part_top5_recall', 0):.3f}")
                print(f"    structure_aware: {r.get('structure_aware_top5_recall', 0):.3f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"strategy": strategy, "status": "ERROR", "error": str(e)})

    print("\n" + "=" * 110)
    print(f"{'Strategy':<22}{'Chunks':>8}{'T1-Kw':>8}{'T5-Kw':>8}{'Fact T5':>9}{'Multi T5':>10}{'Struct T5':>11}")
    print("=" * 110)
    for r in results:
        if r.get("status") == "OK":
            print(f"{r['strategy']:<22}{r['n_chunks']:>8}"
                  f"{r['mean_top1_recall']:>8.3f}{r['mean_top5_recall']:>8.3f}"
                  f"{r.get('factual_top5_recall', 0):>9.3f}"
                  f"{r.get('multi_part_top5_recall', 0):>10.3f}"
                  f"{r.get('structure_aware_top5_recall', 0):>11.3f}")
    print("=" * 110)

    with open("benchmark_results_60q.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results_60q.json")


if __name__ == "__main__":
    main()
