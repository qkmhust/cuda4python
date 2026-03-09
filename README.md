# mini-cuda-llm

A beginner-friendly CUDA demo with Python bindings.

## What This Demo Includes
- CUDA kernel: vector addition (`src/vector_add.cu`)
- Advanced CUDA path: cached memory + async stream (`src/vector_add_advanced.cu`)
- C API wrapper: easy to call from Python (`include/kernels.h`)
- Python package: `ctypes` + NumPy interface (`python/mini_cuda_llm`)
- Validation and benchmark scripts

## Project Structure
- `CMakeLists.txt`: build configuration
- `src/main.cpp`: C++ demo program
- `src/vector_add.cu`: CUDA implementation and host wrapper
- `src/vector_add_advanced.cu`: advanced optimized implementation
- `include/kernels.h`: exported C API declarations
- `python/mini_cuda_llm/validate.py`: correctness checks
- `python/mini_cuda_llm/benchmark.py`: CUDA vs NumPy timing
- `scripts/monitor_gpu.sh`: real-time GPU monitor

## Build And Run (C++)
```bash
cd /root/mini-cuda-llm
cmake -S . -B build
cmake --build build -j
./build/mini_cuda_llm
```

## Install Python Package
```bash
cd /root/mini-cuda-llm
cmake --install build --prefix /root/mini-cuda-llm
python3 -m pip install -e python
```

## Validate
```bash
python3 -m mini_cuda_llm.validate
```

## Benchmark
```bash
python3 -m mini_cuda_llm.benchmark --size 1000000 --rounds 30 --warmup 5
```

This benchmark prints 3 stages:
- `CUDA basic`: beginner-friendly, stateless implementation
- `CUDA advanced`: cached device memory + async stream
- `NumPy`: CPU baseline

## GPU Monitor (watch/night)
```bash
cd /root/mini-cuda-llm
./scripts/monitor_gpu.sh
```

If you prefer non-interactive logging:
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

## API Contract (Rigorous)
- `vectorAddHost(a, b, c, n)`:
  - Inputs must be non-null
  - `n > 0`
  - Returns CUDA error code (`0` means success)
- `validateVectorAdd(n, expected, tolerance)`:
  - `n > 0`, `tolerance >= 0`
  - Returns `0` on pass, `-1` on mismatch, CUDA code on runtime failure
- Python `cuda_vector_add_numpy(a, b)`:
  - Requires same shape, 1D arrays
  - Converts to contiguous `float32`

## Notes For Beginners
- This version favors clarity and strict checks over aggressive optimization.
- Keep `basic` as your reference implementation, then compare with `advanced` step by step.
- If you need more throughput after `advanced`, optimize transfer strategy next (pinned memory, stream overlap, data reuse).
