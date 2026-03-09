# Performance Summary

## VectorAdd
- Best advanced gain over basic: 2.273x at size=100000
- Max advanced bandwidth: 10.964 GB/s

## GEMM
- Best cuBLAS gain over basic: 2.730x at shape=(1024,1024,1024)
- Max cuBLAS throughput: 1857.317 GFLOPS

## Accuracy
- VectorAdd max abs diff (advanced): 0.00000000
- Softmax max abs diff (advanced): 0.00000001
- GEMM max abs diff (cuBLAS): 0.00012207
