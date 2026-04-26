#!/usr/bin/env python3
"""HTML visualization of chunk boundaries for each strategy."""

import pickle
from html import escape
from pathlib import Path


STRATEGIES = [
    ("recursive_sections", "sections", "textbook_index"),
    ("sliding_window", "sliding_window", "textbook_sliding_window"),
    ("sentence_boundary", "sentence_boundary", "textbook_sentence_boundary"),
    ("paragraph", "paragraph", "textbook_paragraph"),
    ("adaptive", "adaptive", "textbook_adaptive"),
]

COLORS = ["#fff4e6", "#e6f7ff", "#f0fff0", "#fff0f5", "#fffacd",
          "#e6e6fa", "#f0f8ff", "#ffefd5", "#f5fffa", "#fce4ec"]


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Chunk Visualization: {strategy}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 950px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.55; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.3em; }}
  .meta {{ background: #f5f5f5; padding: 0.8em 1em; border-radius: 5px; margin-bottom: 1.5em; font-size: 0.9em; }}
  .chunk {{ padding: 0.6em 0.9em; margin: 0.25em 0; border-left: 4px solid #888; border-radius: 3px; white-space: pre-wrap; word-wrap: break-word; font-family: Georgia, serif; font-size: 0.95em; }}
  .chunk-header {{ font-size: 0.75em; color: #666; font-weight: bold; margin-bottom: 0.3em; font-family: -apple-system, sans-serif; }}
  details {{ margin: 1em 0; }}
  summary {{ cursor: pointer; font-weight: bold; padding: 0.5em; background: #f0f0f0; border-radius: 3px; }}
</style></head><body>
<h1>Chunk Boundary Visualization</h1>
<div class="meta">
  <b>Strategy:</b> {strategy}<br>
  <b>Total chunks:</b> {n_chunks:,}<br>
  <b>Avg chunk size:</b> {avg_size:.0f} chars<br>
  <b>Median chunk size:</b> {median_size:.0f} chars<br>
  <b>Showing:</b> first {n_shown} chunks
</div>
<p style="font-size: 0.9em; color: #555;">Each colored block is one chunk.</p>
{chunks_html}
<details><summary>Show all {n_chunks} chunks</summary>{all_chunks_html}</details>
</body></html>"""


def render_chunk(idx, text, max_preview=1500):
    color = COLORS[idx % len(COLORS)]
    clean = text
    if clean.startswith("Description:"):
        marker = " Content: "
        if marker in clean:
            clean = clean.split(marker, 1)[1]
    preview = clean[:max_preview]
    if len(clean) > max_preview:
        preview += f"... [truncated, full {len(clean)} chars]"
    return (f'<div class="chunk" style="background: {color};">'
            f'<div class="chunk-header">Chunk #{idx} &mdash; {len(clean)} chars</div>'
            f'{escape(preview)}</div>')


def visualize_strategy(strategy, folder, prefix, n_preview=30):
    chunks_path = Path("index") / folder / f"{prefix}_chunks.pkl"
    if not chunks_path.exists():
        print(f"  SKIP {strategy}: {chunks_path} not found")
        return False

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    n = len(chunks)
    sizes = [len(c) for c in chunks]
    avg_size = sum(sizes) / n if n else 0
    median_size = sorted(sizes)[n // 2] if n else 0

    n_shown = min(n_preview, n)
    chunks_html = "\n".join(render_chunk(i, chunks[i]) for i in range(n_shown))
    all_chunks_html = "\n".join(render_chunk(i, chunks[i], 400) for i in range(n_shown, n))

    html = HTML_TEMPLATE.format(
        strategy=strategy, n_chunks=n, avg_size=avg_size, median_size=median_size,
        n_shown=n_shown, chunks_html=chunks_html, all_chunks_html=all_chunks_html)

    out_dir = Path("chunk_visualizations")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{strategy}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  {strategy}: {n} chunks -> {out_path}")
    return True


def make_index_page():
    out_dir = Path("chunk_visualizations")
    strategies = [s[0] for s in STRATEGIES]
    links = "\n".join(f'<li><a href="{s}.html">{s}</a></li>' for s in strategies)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Chunk Visualizations</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:3em auto;padding:0 1em}}
ul{{font-size:1.1em;line-height:1.8}}a{{color:#06c;text-decoration:none}}a:hover{{text-decoration:underline}}</style>
</head><body><h1>Chunk Boundary Visualizations</h1>
<ul>{links}</ul></body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\nIndex: {out_dir}/index.html")


def main():
    print("Generating chunk visualizations\n")
    rendered = False
    for strategy, folder, prefix in STRATEGIES:
        if visualize_strategy(strategy, folder, prefix):
            rendered = True
    if rendered:
        make_index_page()


if __name__ == "__main__":
    main()
