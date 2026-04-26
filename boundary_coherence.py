#!/usr/bin/env python3
"""Boundary coherence analysis for chunking strategies."""

import json
import pickle
import re
from pathlib import Path

import numpy as np


STRATEGIES = [
    ("recursive_sections", "sections", "textbook_index"),
    ("sliding_window", "sliding_window", "textbook_sliding_window"),
    ("sentence_boundary", "sentence_boundary", "textbook_sentence_boundary"),
    ("paragraph", "paragraph", "textbook_paragraph"),
    ("adaptive", "adaptive", "textbook_adaptive"),
]


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def load_chunks(folder, prefix):
    p = Path("index") / folder / f"{prefix}_chunks.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def analyze(strategy, folder, prefix, model, max_pairs=200):
    chunks = load_chunks(folder, prefix)
    if chunks is None:
        return {"strategy": strategy, "status": "MISSING"}

    n = len(chunks)
    pairs = []
    for i in range(min(n - 1, max_pairs)):
        cs = split_sentences(chunks[i])
        ns = split_sentences(chunks[i + 1])
        if cs and ns:
            pairs.append((cs[-1], ns[0]))

    if not pairs:
        return {"strategy": strategy, "status": "NO_PAIRS", "n_chunks": n}

    last_sents = [p[0] for p in pairs]
    first_sents = [p[1] for p in pairs]
    last_e = model.encode(last_sents, show_progress_bar=False, convert_to_numpy=True)
    first_e = model.encode(first_sents, show_progress_bar=False, convert_to_numpy=True)

    sims = [cosine_sim(last_e[i], first_e[i]) for i in range(len(pairs))]

    return {
        "strategy": strategy,
        "status": "OK",
        "n_chunks": n,
        "n_pairs_analyzed": len(pairs),
        "mean_boundary_similarity": float(np.mean(sims)),
        "median_boundary_similarity": float(np.median(sims)),
        "std_boundary_similarity": float(np.std(sims)),
    }


def main():
    print("Loading sentence-transformer")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nAnalyzing boundary coherence\n")
    results = []
    for strategy, folder, prefix in STRATEGIES:
        print(f"  {strategy}...", end=" ", flush=True)
        r = analyze(strategy, folder, prefix, model)
        results.append(r)
        if r["status"] == "OK":
            print(f"{r['n_chunks']} chunks, mean_sim={r['mean_boundary_similarity']:.4f}")
        else:
            print(r["status"])

    print("\n" + "=" * 80)
    print(f"{'Strategy':<22}{'Chunks':>8}{'Mean Sim':>12}{'Median':>10}{'Std':>10}")
    print("=" * 80)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['strategy']:<22}{r['n_chunks']:>8}"
                  f"{r['mean_boundary_similarity']:>12.4f}"
                  f"{r['median_boundary_similarity']:>10.4f}"
                  f"{r['std_boundary_similarity']:>10.4f}")
    print("=" * 80)

    with open("boundary_coherence_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to boundary_coherence_results.json")


if __name__ == "__main__":
    main()
