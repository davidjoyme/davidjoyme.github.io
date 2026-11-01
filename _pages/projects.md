---
layout: archive
title: "Projects (scroll to see more)"
permalink: /projects/
author_profile: true
---

## Mini-Inference — [GitHub](https://github.com/davidjoyme)

*Python, CUDA C++, Triton, PyTorch*

- Built an LLM inference engine for MoE (OLMoE),Qwen3 models, with continuous batching, paged KV cache, prefix caching, chunked prefill and FP8/INT8/INT4 quantization mimicking serving engines SGLang and vLLM.
- Raised throughput 2.3× (1,521 to 3,447 tok/s) using profile-driven optimization: fused-MoE Triton and CUDA C++ kernels, a paged-attention kernel at 92% of peak bandwidth, and CUDA graphs.
- Profiled a dense-model decode bottleneck to cuBLAS kernel selection with a custom GEMM benchmark fused projections with autotuned weight layouts, saving 2 ms per step as predicted.

---

## FlashDecode-CUDA — [GitHub](https://github.com/davidjoyme)

*CUDA C++ , Nsight Compute , PyTorch , Python*

- Designed a paged GQA decode-attention CUDA kernel for RTX 5090 GPU achieving 6.2–6.7x faster than PyTorch Attention implementation at working sets beyond L2, via split-KV decomposition and 16-byte vectorized loads.
- Built a Qwen2.5-1.5B decode engine with a paged KV cache and fused RMSNorm kernel, validated bit-exact against HuggingFace across all 28 layers.

---

## Kiln ML Compiler — [GitHub](https://github.com/davidjoyme)

*Python, MLIR, Triton, CUDA*

- Built Kiln, an MLIR-based PyTorch compiler for CPU and GPU (Triton); its fused kernels run 1.7–2.3x faster than eager PyTorch.
- Achieved 16\% more TFLOPs on an RTX 5090 with autotuned Triton matmuls; a GPT-2 block runs 1.2–1.9x faster than PyTorch, matching Pytorch's compile modules.
- Implemented SM-aware tiling which raised matmul throughput 5x (6.6 → 34.9 TFLOP/s).