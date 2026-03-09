#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/mini-cuda-llm"
OUT_DIR="${1:-/root/mini-cuda-llm/reports/latest}"
ROUNDS="${2:-10}"
WARMUP="${3:-3}"

cd "$ROOT_DIR"
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix /root/mini-cuda-llm
python3 -m pip install -e python
python3 -m mini_cuda_llm.perf_pipeline --out "$OUT_DIR" --rounds "$ROUNDS" --warmup "$WARMUP"

echo "Done. Open: $OUT_DIR/summary.md and $OUT_DIR/performance_overview.png"
