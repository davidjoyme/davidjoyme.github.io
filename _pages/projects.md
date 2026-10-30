---
layout: archive
title: "Projects (scroll to see more)"
permalink: /projects/
author_profile: true
---

## Mini-Inference — [GitHub](https://github.com/davidjoyme)

*Inference Engineering Project*

- Built an LLM inference engine from scratch for MoE (OLMoE-1B-7B) and dense (Qwen3-4B) models: continuous batching, paged KV cache, prefix caching and FP8/INT4 quantization, verified token-exact by 42 tests.
- Raised throughput 2.3× (1,521 → 3,447 tok/s) with custom Triton and CUDA C++ kernels: fused MoE (37 → 5 launches per layer), paged attention at 92% of peak bandwidth, and CUDA graphs (1.9× lower latency at batch 1).
- Profiled a dense-model decode bottleneck to cuBLAS kernel selection with a custom GEMM benchmark (skinny matmuls at 17–50% of peak bandwidth); fused projections with autotuned weight layouts, saving 2 ms per step as predicted.

---

## FlashDecode-CUDA — [GitHub](https://github.com/davidjoyme)

*Kernel Engineering Project*

- Wrote a paged GQA decode-attention CUDA kernel for RTX 5090 (sm_120), an architecture with no FlashAttention, FlashInfer, or FlashMLA backend; 6.2–6.7× faster than PyTorch SDPA at working sets beyond L2, via split-KV decomposition and 16-byte vectorized loads (48 registers, 0 spills).
- Built a Qwen2.5-1.5B decode engine with a paged KV cache and fused RMSNorm kernel, validated bit-exact against HuggingFace across all 28 layers (32/32 activations at 0.0 error, 32/32 greedy tokens); 38-test suite mutation-tested against 9 injected bug classes.
- Profiled with Nsight Compute to drive three kernel iterations: v0 was 8× slower than SDPA (latency-bound on 10,240 dependent warp shuffles per head), v1 cut that to 4,096 for 2.9×, and v2’s split-KV gave a further 11× — ending at 20% of measured HBM peak with the remaining gap identified. Caught two bugs by test design rather than luck: a NaN from -inf − -inf in empty split partials, found by sweeping split counts; and a .item() device sync in the host wrapper costing up to 23.8× more than the kernel it wrapped.

---

## Kiln ML Compiler — [GitHub](https://github.com/davidjoyme)

*Machine Learning Compiler Project*

- Built Kiln, an MLIR-based PyTorch compiler for CPU and GPU (Triton); its fused kernels run 1.7–2.3x faster than eager PyTorch.
- Reached 49 TFLOP/s (cuBLAS: 42) on an RTX 5090 with autotuned Triton matmuls; GPT-2 block runs 1.2–1.9x faster than PyTorch, matching torch.compile.
- 171 tests and fuzzing over 120 random graphs; SM-aware tiling raised matmul throughput 5x (6.6 → 34.9 TFLOP/s).