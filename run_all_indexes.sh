#!/bin/bash
STRATEGIES=("sliding_window" "sentence_boundary" "paragraph" "adaptive")

for strat in "${STRATEGIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Indexing with chunk_mode=$strat"
    echo "============================================================"
    sed -i '' "s/^chunk_mode: .*/chunk_mode: \"$strat\"/" config/config.yaml
    python -m src.main index --index_prefix "textbook_${strat}"
    echo "Done with $strat"
done

sed -i '' "s/^chunk_mode: .*/chunk_mode: \"recursive_sections\"/" config/config.yaml
echo "All strategies indexed!"
