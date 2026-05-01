#!/usr/bin/env python3
"""Ablation study on the adaptive chunker's routing thresholds.

Re-runs the adaptive routing logic on real documents at varying
paragraph_density thresholds and reports which fixed strategy is
selected and how many chunks result.
"""

import json
import re
from pathlib import Path

from src.preprocessing.chunking import (
    AdaptiveConfig,
    AdaptiveStrategy,
    ParagraphAwareStrategy,
    SentenceBoundaryStrategy,
)


THRESHOLDS = [1.0, 2.0, 3.0, 5.0]


def analyze_document(text: str) -> dict:
    paragraphs = ParagraphAwareStrategy.PARAGRAPH_RE.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    sentences = SentenceBoundaryStrategy.SENTENCE_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total_len = len(text) if text else 1
    return {
        "num_paragraphs": len(paragraphs),
        "num_sentences": len(sentences),
        "avg_paragraph_len": total_len / max(len(paragraphs), 1),
        "avg_sentence_len": total_len / max(len(sentences), 1),
        "paragraph_density": len(paragraphs) / (total_len / 1000),
        "has_headers": bool(re.search(r"^#{1,6}\s", text, re.MULTILINE)),
        "total_length": total_len,
    }


def select_strategy_name(features: dict, paragraph_density_threshold: float) -> str:
    """Reimplement the routing logic with a configurable threshold."""
    if features["paragraph_density"] > paragraph_density_threshold or features["has_headers"]:
        return "paragraph"
    if features["num_sentences"] > 5 and features["avg_sentence_len"] < 500:
        return "sentence_boundary"
    return "sliding_window"


SAMPLE_DOCS = {
    "textbook_dense_prose": (
        "Concurrency control is a fundamental component of any modern database management "
        "system that supports concurrent transactions. The goal of concurrency control is "
        "to ensure that the execution of concurrent transactions yields a result equivalent "
        "to some serial execution of those transactions. Without concurrency control, "
        "concurrent updates to shared data can produce inconsistent or incorrect results. "
        "There are several approaches to concurrency control, including lock-based "
        "protocols, timestamp-based protocols, and validation-based protocols. The two-phase "
        "locking protocol is the most widely used lock-based approach. Under two-phase "
        "locking, transactions are divided into two phases: a growing phase during which "
        "locks are acquired but not released, and a shrinking phase during which locks are "
        "released but not acquired. This guarantees conflict-serializable schedules but does "
        "not prevent deadlocks. Various deadlock detection and prevention mechanisms are "
        "used in practice, such as wait-for graphs and timeouts."
    ),
    "structured_with_headers": (
        "## Concurrency Control\n\n"
        "Concurrency control ensures correctness in multi-transaction systems.\n\n"
        "## Two-Phase Locking\n\n"
        "Two-phase locking has a growing phase and a shrinking phase. The protocol "
        "guarantees serializability.\n\n"
        "## Deadlocks\n\n"
        "Deadlocks can occur under two-phase locking. They are detected via wait-for "
        "graphs or prevented via timeouts."
    ),
    "short_choppy_sentences": (
        "Locks protect data. Locks are shared or exclusive. Shared locks allow reads. "
        "Exclusive locks allow writes. Two transactions cannot both hold exclusive locks. "
        "But two transactions can both hold shared locks. Locks are released at commit time. "
        "This is called strict two-phase locking. Strict 2PL prevents cascading rollbacks. "
        "It is widely used in commercial systems."
    ),
    "many_short_paragraphs": (
        "Locks protect data.\n\n"
        "Shared locks allow reads.\n\n"
        "Exclusive locks allow writes.\n\n"
        "Two transactions cannot both hold exclusive locks.\n\n"
        "But two transactions can both hold shared locks.\n\n"
        "Locks are released at commit time.\n\n"
        "This is called strict two-phase locking.\n\n"
        "Strict 2PL prevents cascading rollbacks.\n\n"
    ),
}


def chunk_with_adaptive(text: str) -> list:
    cfg = AdaptiveConfig(max_chunk_size=2000, overlap=200)
    strategy = AdaptiveStrategy(cfg)
    return strategy.chunk(text)


def main():
    print("Adaptive chunker threshold ablation")
    print(f"Thresholds tested: {THRESHOLDS}\n")

    n_chunks_default = {}
    for doc_name, doc_text in SAMPLE_DOCS.items():
        chunks = chunk_with_adaptive(doc_text)
        n_chunks_default[doc_name] = len(chunks)

    results = {}
    for doc_name, doc_text in SAMPLE_DOCS.items():
        print(f"=== {doc_name} ({len(doc_text)} chars) ===")
        features = analyze_document(doc_text)
        print(f"  features: density={features['paragraph_density']:.2f} "
              f"sentences={features['num_sentences']} "
              f"avg_sent_len={features['avg_sentence_len']:.0f} "
              f"has_headers={features['has_headers']}")

        doc_results = []
        for thr in THRESHOLDS:
            chosen = select_strategy_name(features, thr)
            doc_results.append({
                "threshold": thr,
                "chosen_strategy": chosen,
                "features": features,
            })
            print(f"  threshold={thr:>4}  -> {chosen}")
        results[doc_name] = {
            "default_n_chunks": n_chunks_default[doc_name],
            "features": features,
            "ablation": doc_results,
        }
        print()

    print("=" * 100)
    print(f"{'Document':<32}", end="")
    for thr in THRESHOLDS:
        print(f"  thr={thr:<5}", end="")
    print()
    print("-" * 100)
    for doc_name, doc_r in results.items():
        print(f"{doc_name:<32}", end="")
        for r in doc_r["ablation"]:
            print(f"  {r['chosen_strategy']:<8}", end="")
        print()
    print("=" * 100)

    out_path = Path("ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
