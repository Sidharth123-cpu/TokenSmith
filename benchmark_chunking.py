#!/usr/bin/env python3
"""Retrieval benchmark for chunking strategies on the Silberschatz textbook."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import faiss

from src.embedder import SentenceTransformer


QUESTIONS = [
    {"q": "What problem does concurrency control solve in a database system?",
     "keywords": ["concurrent", "consistency", "transactions", "isolation"],
     "section_fragment": "Transaction Isolation"},
    {"q": "What is the two-phase locking protocol?",
     "keywords": ["growing phase", "shrinking phase", "two-phase", "locking"],
     "section_fragment": "Two-Phase Locking"},
    {"q": "What is the difference between strict two-phase locking and basic two-phase locking?",
     "keywords": ["strict", "exclusive", "commits", "cascading"],
     "section_fragment": "Two-Phase Locking"},
    {"q": "How are deadlocks detected in database systems?",
     "keywords": ["deadlock", "wait-for graph", "cycle", "detection"],
     "section_fragment": "Deadlock"},
    {"q": "What is timestamp-based concurrency control?",
     "keywords": ["timestamp", "ordering", "read", "write"],
     "section_fragment": "Timestamp"},
    {"q": "What is snapshot isolation?",
     "keywords": ["snapshot", "version", "read", "isolation"],
     "section_fragment": "Snapshot"},
    {"q": "How does ARIES recovery work?",
     "keywords": ["analysis", "redo", "undo", "log"],
     "section_fragment": "ARIES"},
    {"q": "What is write-ahead logging?",
     "keywords": ["log", "before", "disk", "write-ahead"],
     "section_fragment": "Recovery"},
    {"q": "What is a B+ tree index used for?",
     "keywords": ["B+", "tree", "leaf", "search"],
     "section_fragment": "B+"},
    {"q": "How does the buffer manager handle page replacement?",
     "keywords": ["buffer", "page", "replacement", "LRU"],
     "section_fragment": "Buffer"},
    {"q": "What are the ACID properties of a transaction?",
     "keywords": ["atomicity", "consistency", "isolation", "durability"],
     "section_fragment": "Transaction"},
    {"q": "What is query optimization?",
     "keywords": ["query", "plan", "cost", "optimizer"],
     "section_fragment": "Query Optimization"},
    {"q": "What is a hash join?",
     "keywords": ["hash", "join", "partition", "build"],
     "section_fragment": "Hash"},
    {"q": "What is normalization in database design?",
     "keywords": ["normal form", "dependency", "decomposition"],
     "section_fragment": "Normal"},
    {"q": "What is a foreign key constraint?",
     "keywords": ["foreign key", "reference", "constraint"],
     "section_fragment": "Foreign"},
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
        print(f"  [{i+1}/{len(queries)}] norm={norm:.2f}")
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
            "top1_recall": keyword_recall(retrieved_chunks[0], qd["keywords"]),
            "top5_recall": keyword_recall(" ".join(retrieved_chunks[:5]), qd["keywords"]),
            "top1_section_hit": section_hit(retrieved_meta[:1], qd["section_fragment"]),
            "top5_section_hit": section_hit(retrieved_meta[:5], qd["section_fragment"]),
        })

    return {
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
        "per_question": per_q,
    }


def main():
    embed_model = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    print(f"Loading {embed_model}\n")
    embedder = SentenceTransformer(embed_model)

    queries = [q["q"] for q in QUESTIONS]
    print(f"Embedding {len(queries)} queries")
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
                print(f"  T1 kw: {r['mean_top1_recall']:.3f} | T5 kw: {r['mean_top5_recall']:.3f}")
                print(f"  T1 sec: {r['top1_section_hit_rate']:.3f} | T5 sec: {r['top5_section_hit_rate']:.3f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"strategy": strategy, "status": "ERROR", "error": str(e)})

    print("\n" + "=" * 100)
    print(f"{'Strategy':<22}{'Chunks':>8}{'AvgLen':>9}{'T1-Kw':>8}{'T5-Kw':>8}{'T1-Sec':>9}{'T5-Sec':>9}")
    print("=" * 100)
    for r in results:
        if r.get("status") == "OK":
            print(f"{r['strategy']:<22}{r['n_chunks']:>8}{r['avg_chunk_size']:>9.0f}"
                  f"{r['mean_top1_recall']:>8.3f}{r['mean_top5_recall']:>8.3f}"
                  f"{r['top1_section_hit_rate']:>9.3f}{r['top5_section_hit_rate']:>9.3f}")
    print("=" * 100)

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
