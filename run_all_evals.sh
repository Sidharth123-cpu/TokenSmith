#!/bin/bash
set -e

echo "Running evaluation pipeline"
echo ""

echo "Checking indexes..."
for entry in "sections/textbook_index" "sliding_window/textbook_sliding_window" "sentence_boundary/textbook_sentence_boundary" "paragraph/textbook_paragraph" "adaptive/textbook_adaptive"; do
    f="index/${entry}_chunks.pkl"
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "1/3: Retrieval benchmark"
python benchmark_chunking.py

echo ""
echo "2/3: Boundary coherence analysis"
python boundary_coherence.py

echo ""
echo "3/3: Chunk visualizations"
python visualize_chunks.py

echo ""
echo "Done"
