#!/usr/bin/env bash
set -euo pipefail

# Real-time GPU monitor. Press Ctrl+C to stop.
watch -n 1 nvidia-smi
