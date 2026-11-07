# 05 Ampere sm86 (RTX 3090) as measured

Scope: 2× RTX 3090 (compute capability 8.6), 250 W power limit, torch 2.14.0+cu132,
Triton 3.8.0 on 0.33 (3.7.1 on 0.28), FlashInfer 0.7.2.post1, AWQ-INT4 target run by
Marlin W4A16.

## fp8 on sm86

- sm86 has no fp8 tensor instructions; fp8 KV on this GPU is a storage format that a
  kernel decodes before the math.
- Triton 3.7.1 on sm86 fails to compile both loads and stores of `fp8e4nv`
  (`CompilationError`, our kernel test, 2026-11-03), and vLLM's Triton attention backend
  refuses fp8 KV below SM89 (`native FP8 (fp8e4nv) requires SM89+`).
- FlashInfer serves fp8 KV on sm86; the CUDA op `reshape_and_cache_flash` writes it.
- E003 keeps fp8 bytes in the cache and decodes them in Triton with bit operations: the
  e4m3fn sign, exponent and mantissa go into fp16 bits, a multiply by 2^8 fixes the
  exponent bias (15 − 7 = 8), then the tensor scale applies. Every finite e4m3fn value is
  exact in fp16. Bytes 0x7F and 0xFF are NaN (a shortcut decoder turns them into ±480).
  The trick is wrong for e5m2.
- Any fp8 tensor reaching a Triton kernel on sm86 fails at start (`type fp8e4nv not
  supported in this architecture`); our first E003 start failed this way because the
  query was quantized to fp8. Keep the query in bf16.
- E003 accuracy (kernel test): attention output vs a bf16 reference differs by at most
  2.4e-4 (causal) and 7.8e-3 (non-causal), from tile size 32 vs 16 accumulation order.

## Small-M GEMMs

The GDN input projection `in_proj_ba` stays bf16 (the AWQ checkpoint leaves it
unquantized), shape N = 48, K = 5120 per GPU at TP2. During spec decode M = 8 tokens per
request.

| M | Kernel cuBLAS picks (nsys) | Time |
|---|---|---|
| 1 (no spec) | gemv + split-K reduce pair | 5.6 µs per call (0.28), 5.87 µs (0.33) |
| 8 (1 request, spec) | `cutlass_80_wmma_tensorop_bf16_s161616gemm` 16×16, grid of 4 CTAs | 34.7 µs per call isolated, 38.1 µs in the trace; 1,818 µs per step on 0.33 |
| 32 (4 requests, spec) | another cuBLAS kernel | 0.44 ms per step (0.28) |

- E006a replaces M ≤ 16 with a Triton split-K bf16 GEMM; max abs error vs fp32 0.4999
  at M=8 (`F.linear`: 0.5000). E006c makes it dispatch at run time (see
  [06-torch-compile-cuda-graphs.md](06-torch-compile-cuda-graphs.md)). With both, the
  `in_proj_ba` class is 149.7 µs per step (E022) and the step 1.3–1.6 ms shorter (E011).
- Look for this pattern at any small-M bf16 linear that a quantized checkpoint leaves
  unquantized: profile at the real spec-decode M, not at M=1.

## Marlin and other int4 kernels

- Marlin W4A16 at M=8 costs what it costs at M=1: 9.82 vs 9.73 ms per step for all
  target Marlin calls, about 77% of memory bandwidth (0.28 nsys, EXP-0). Verifying 8
  tokens is nearly free in the large GEMMs.
- Per matrix on 0.33 (E022): 503–756 GB/s at one request (M=8); at four requests (M=32)
  each call takes 47–76% longer and reaches 37.0–53.9 TFLOPS. Table:
  [step-anatomy.md](../../../qwen3.8-27b/results/step-anatomy.md#all-reduce-and-marlin-e022).
- Estimate, not measured: the datasheet FP16/BF16 tensor rate with FP32 accumulate is
  71 TFLOPS at 1,695 MHz, 65.3 TFLOPS at the 1,560 MHz we see under load, so M=32 Marlin
  calls run at 57–83% of that.
- The bf16 lm_head GEMM (1,271,398,400 B per GPU) runs at 860 GB/s, 1.48 ms.
- Other int4 linear kernels vLLM 0.33 lists for CUDA: CutlassW4A8, Machete, Marlin,
  Conch, Exllama, TritonW4A16, Humming. Measured on our checkpoint (E023): TritonW4A16
  47.11–47.66 ms per step, Humming 19.83–20.18 ms, Marlin 19.75–19.93 ms. Exllama refuses
  bf16 activations; the conch package was not installed.
- Marlin with int8 activations (`VLLM_MARLIN_INPUT_DTYPE=int8`) asserts
  `W8A8 is not supported by marlin kernel.` at start for our asymmetric uint4 checkpoint
  with zero points; it needs symmetric `uint4b8` weights (E024).
- GDN recurrence: 0.33's fused GDN MTP decode kernel applies when
  `num_v_heads // num_k_heads` is in (1, 2, 3, 4, 8); 48 // 16 = 3 passes. It takes 1.29 ms
  per step in the target (E022). `VLLM_GDN_DECODE_KERNEL=triton` shortened the step by
  0.3–0.6 ms at KL 4.54e-4 (E017) and did not stack with PCIe IPC (E023).

## Power and clocks

| Condition | SM clock | Power | Source |
|---|---|---|---|
| 0.28 spec, 1 request | 1,766 MHz | 248 W | EXP-0 |
| 0.28 spec, 4 requests | 1,682 MHz | 236 W | EXP-0 |
| 0.28 no spec, 1 request | 1,875 MHz | 250 W | EXP-0 |
| 0.33 production code, 1 and 4 requests mixed | 1,575 (GPU0) / 1,545 (GPU1) MHz median | 249 W median | E022 |

- The throttle reason under decode load is the software power cap (0x4) against our
  250 W limit. The memory clock under CUDA load is 9,501 MHz (P2 state), not 9,751 MHz.
- GPU utilization on 0.28 spec decode at one request was 77% (nvidia-smi), 98% without
  speculation.
- Power limit sweep on 0.28 (one request): 250 / 225 / 200 W gave SM 1,823 / 1,686 /
  1,530 MHz and code steps 29.0 / 27.8 / 29.5 ms, prose 27.9 / 27.9 / 29.8 ms.
- Not tried: unlocking the GeForce half-rate FP16/BF16-with-FP32-accumulate tensor mode,
  power limits above 250 W, and locked memory clocks.
