#!/usr/bin/env python3
"""Generate plots from existing benchmark and boundary coherence results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRATEGY_COLORS = {
    "recursive_sections": "#4c78a8",
    "sliding_window": "#f58518",
    "sentence_boundary": "#e45756",
    "paragraph": "#72b7b2",
    "adaptive": "#54a24b",
}

LABEL_OFFSETS = {
    "recursive_sections": (12, 8),
    "sliding_window": (12, 8),
    "sentence_boundary": (12, 14),
    "paragraph": (12, -16),
    "adaptive": (12, -16),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_per_question_breakdown(bench, out_path):
    strategies = [r["strategy"] for r in bench if r.get("status") == "OK"]
    n_questions = len(bench[0]["per_question"])
    matrix = np.zeros((len(strategies), n_questions))
    for i, r in enumerate(bench):
        if r.get("status") != "OK":
            continue
        for j, q in enumerate(r["per_question"]):
            matrix[i, j] = q["top5_recall"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xticks(range(n_questions))
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_questions)], rotation=0, fontsize=8)
    ax.set_xlabel("Question")
    ax.set_title("Top-5 Keyword Recall per Question per Strategy")
    plt.colorbar(im, ax=ax, label="Recall")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_winners_per_question(bench, out_path):
    strategies = [r["strategy"] for r in bench if r.get("status") == "OK"]
    n_questions = len(bench[0]["per_question"])
    wins = {s: 0 for s in strategies}
    for j in range(n_questions):
        scores = {r["strategy"]: r["per_question"][j]["top5_recall"]
                  for r in bench if r.get("status") == "OK"}
        max_score = max(scores.values())
        winners = [s for s, sc in scores.items() if sc == max_score]
        for w in winners:
            wins[w] += 1 / len(winners)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(wins.keys())
    counts = [wins[n] for n in names]
    colors = [STRATEGY_COLORS[n] for n in names]
    bars = ax.bar(names, counts, color=colors, edgecolor="#333", linewidth=0.8)
    ax.set_ylabel("Questions won (ties shared fractionally)")
    ax.set_title(f"Per-question winners across {n_questions} questions (Top-5 keyword recall)")
    ax.tick_params(axis="x", rotation=18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{count:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_coherence_vs_retrieval(bench, coh, out_path):
    coh_by_strat = {r["strategy"]: r for r in coh if r.get("status") == "OK"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    points = []
    for r in bench:
        if r.get("status") != "OK":
            continue
        strat = r["strategy"]
        c = coh_by_strat.get(strat)
        if not c:
            continue
        x = c["mean_boundary_similarity"]
        y = r["mean_top5_recall"]
        points.append((strat, x, y))

    for strat, x, y in points:
        color = STRATEGY_COLORS[strat]
        ax.scatter(x, y, s=220, alpha=0.9, color=color,
                   edgecolor="#222", linewidth=1.2, zorder=3)
        dx, dy = LABEL_OFFSETS[strat]
        ax.annotate(strat, (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10.5,
                    fontweight="medium")

    ax.set_xlabel("Mean boundary similarity (lower = cleaner topic shifts)",
                  fontsize=11)
    ax.set_ylabel("Mean top-5 keyword recall", fontsize=11)
    ax.set_title("Boundary Coherence vs. Retrieval Quality",
                 fontsize=12.5, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    xmin = min(p[1] for p in points) - 0.05
    xmax = max(p[1] for p in points) + 0.08
    ymin = min(p[2] for p in points) - 0.025
    ymax = max(p[2] for p in points) + 0.025
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    annotation = ("Cleaner boundaries (left) do NOT predict\n"
                  "better retrieval (top). Sentence-boundary\n"
                  "and adaptive strategies retrieve well\n"
                  "despite worst boundary scores.")
    ax.text(0.98, 0.04, annotation,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff8e1",
                      edgecolor="#c9a227", linewidth=1, alpha=0.95))

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def main():
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    bench = load_json("benchmark_results.json")
    coh = load_json("boundary_coherence_results.json")

    print("Generating plots...")
    plot_per_question_breakdown(bench, out_dir / "per_question_heatmap.png")
    plot_winners_per_question(bench, out_dir / "per_question_winners.png")
    plot_coherence_vs_retrieval(bench, coh, out_dir / "coherence_vs_retrieval.png")
    print("Done.")


if __name__ == "__main__":
    main()
