# Decode step anatomy

Where the time of one speculative decode step goes for Qwen3.8-27B + DFlash2 (7
speculative tokens, so 8 verified tokens per request per step) on 2× RTX 3090, tensor
parallel 2. Rig and flags: [model README](../README.md#test-rig). Experiment list:
[results](README.md). Method and traps: [measurement skill](../../skills/measurement/SKILL.md).

The model has 64 layers in 16 repeats of (3 GDN `linear_attention` layers + 1
`full_attention` layer); vocabulary 248,320 tokens. Every target layer does two all-reduces (after its output
projection and after the MLP down projection); a step does 139 (target 128, drafter 11).

## One step on vLLM 0.33 with all patches (E022)

Measured 2026-11-04 on the production code (vLLM 0.33.0 + E003, E005, E006a/b/c, before
PCIe IPC all-reduce was switched on), a test instance, nsys with
`--cuda-graph-trace=node`, one prose request of 200 tokens. A GPU cycle runs from the end
of the previous step's drafter lm_head to the end of this step's drafter lm_head; median
of 126 cycles on GPU0.

| Phase | ms | Content |
|---|---|---|
| Prep | 1.82 | kernels 0.20; GPU without work 1.61 (largest gap 1.00 ms, after `_prepare_rope_positions_kernel` and before the first RMSNorm of the target graph; the rest are 50–123 µs gaps between small host-to-device copies) |
| Target, 64 layers | 14.22 | Marlin 9.92, GDN recurrence 1.29, all-reduce 1.21, RMSNorm 0.48, attention 0.38, other 0.30, conv1d 0.19, `in_proj_ba` 0.15, activation 0.15; gaps between kernels 0.04 in total |
| Target lm_head | 1.48 | bf16, 1,271,398,400 B per GPU, 860 GB/s |
| Verify | 0.21 | logits all-gather 0.12 |
| Drafter, 5 layers | 1.37 | Marlin 0.77, bf16 GEMM 0.24, all-reduce 0.11 |
| Drafter lm_head | 1.47 | the drafter reads the target's full lm_head again |
| Cycle | 20.61 | |

- One GDN layer is 15 kernels, 224.8 µs; one attention layer is 14 kernels, 215.0 µs.
  A GDN layer opens with small kernels: `in_proj_ba` 3.10, a reduce 4.93, a bf16 copy
  1.38, zeros 1.09, conv1d 3.78 µs.
- The GDN output projection is int4 in 47 layers and bf16 in layer 0 (the quantization
  ignore list).
- The trace inflates the prep phase. Without a profiler the step is 19.75–19.89 ms (E017);
  the traced cycle has 18.94 ms of GPU work. Estimate, not measured: 0.8–0.95 ms of GPU
  idle per step without a profiler (19.75–19.89 minus 18.94, assuming tracing leaves kernel
  durations unchanged).
- Four concurrent prose requests (per step, from CPU NVTX windows, 95 steps): wall 31.45
  ms, Marlin 17.30, bf16 GEMM including lm_head 3.96, all-reduce 3.32, GDN recurrence
  2.56, GPU idle 1.98 ms.

## CPU side of the same step (E020, E026)

E026 (2026-11-04) timed the TP rank 0 worker with in-process timestamps and no profiler,
on the production code plus PCIe IPC all-reduce, CPU governor `performance`. Medians over
1,500 steps (1 request) and 465 steps (4 requests); rank 1 is within 30 µs of rank 0 on
every segment.

| Segment | 1 request, µs | 4 requests, µs |
|---|---|---|
| Wait inside FlashInfer `seq_lens.cpu()` (target metadata build) | 16,381.4 | 27,105.0 |
| FlashInfer metadata after the wait | 445.3 | 585.7 |
| Embedding inputs | 247.7 | 303.9 |
| Model inputs | 59.6 | 68.1 |
| CUDA graph launch call | 57.9 | 74.4 |
| After the wait to end of graph launch, total | 816.3 | 1,036.4 |
| GPU clock: post-drafter event to pre-replay event | 875.5 | 1,100.8 |
| Engine step on the CPU clock | 19,410.2 | 30,494.1 |

- The worker enters the next `execute_model` while the GPU still runs the previous target
  and drafter. Work before the `seq_lens.cpu()` wait overlaps GPU execution; the 816 µs
  after it is on the critical path.
- The graph launch call is 58 µs without a profiler, 954 µs under nsys node tracing (E020)
  and 871 µs under the torch profiler (E020). Profilers move time into the launch.
- The torch profiler (E020, 50 cycles) attributes, per cycle and inflated by profiling:
  graph replay 949 µs, FlashInfer metadata `build` 817 µs (of which
  `_compute_flashinfer_kv_metadata` 238 µs and `fast_plan_decode` 196 µs),
  `prepare_inputs_embeds` 260 µs, `Tensor.cpu` 227 µs, `pin_memory` 79 µs. Use the
  proportions; take absolute values from the unprofiled table above.
- Other CPU load on the machine lengthens the step: 19.9 → 22.5 ms in one run with
  competing jobs (E021). The CPU is a 6-core / 12-thread desktop part; during decode the two
  TP workers and the engine core each keep about one thread busy.

## All-reduce and Marlin (E022)

GPU0 and GPU1 sit behind one PCIe switch (`nvidia-smi topo` PXB), PCIe gen4 x16, P2P OK.
All-reduce kernels of the two GPUs are paired by order; the table takes the shorter of
the two durations (the later-arriving GPU's), which removes waiting for the other GPU.

| | custom all-reduce (`cross_device_reduce_1stage`) | FlashInfer PCIe IPC (`ipc_tp2_remote_push_kernel`) |
|---|---|---|
| 1 request, 80 KB per call | 8.19 µs (p10 8.06, p90 8.35) | 3.30 µs (p10 3.04, p90 4.86) |
| 4 requests, 320 KB per call | 19.36 µs | 13.28 µs |
| Two-point fit | 22.0 GB/s, intercept 4.5 µs | 24.6 GB/s, intercept about 0 |
| Start skew / end difference, 1 request, medians | 0.92 / 0.10 µs | 2.82 / 2.63 µs |
| All-reduce per step on GPU0, 1 / 4 requests | 1.34 / 3.32 ms | 1.05 / 2.62 ms |
| GPU cycle median, 1 request | 20.61 ms | 20.18 ms |

Without nsys the PCIe IPC step is 19.32–19.52 ms against 19.75–19.93 ms (E017).

Marlin W4A16 per matrix and GPU (weight bytes from the safetensors headers, including
scales and zero points, group size 32):

| Matrix | MB | 1 request (M=8), µs | GB/s | 4 requests (M=32), µs | GB/s | TFLOPS |
|---|---|---|---|---|---|---|
| GDN qkvz | 24.2 | 34.6 | 700 | 56.3 | 430 | 47.7 |
| GDN out | 9.09 | 18.1 | 503 | 26.7 | 341 | 37.7 |
| MLP gate_up | 51.5 | 68.2 | 756 | 105.9 | 486 | 53.9 |
| MLP down | 25.8 | 35.5 | 727 | 62.4 | 414 | 45.7 |
| Attention qkv + gate | 21.2 | 30.9 | 686 | 51.6 | 411 | 45.5 |
| Attention o | 9.09 | 16.2 | 562 | 27.2 | 334 | 37.0 |

- At 4 requests each Marlin call takes 47–76% longer than at 1 request with the same
  weights (ratios of the µs columns above).
- Under load (nvidia-smi every 200 ms): SM clock median 1,575 MHz (GPU0) and 1,545 MHz
  (GPU1), memory 9,501 MHz, power median 249 W against the 250 W limit, throttle reason
  0x4 (software power cap).
- Estimate, not measured: the RTX 3090 datasheet gives 71 TFLOPS dense FP16/BF16 tensor
  with FP32 accumulate at 1,695 MHz, 65.3 TFLOPS scaled to 1,560 MHz; the 4-request Marlin
  calls reach 37.0–53.9 TFLOPS.

## Earlier: vLLM 0.28.0 production (EXP-0, E002)

Measured 2026-11-03 on vLLM 0.28.0 in production (FlashInfer for target and drafter, fp8
KV, LMCache connected), one request, temperature 0.

| Prompt type | Tokens per step | Acceptance | Step, ms | tok/s |
|---|---|---|---|---|
| code | 6.51 | 78.7% | 27.3–27.8 | 233–238 |
| numeric | 5.44 | 63.5% | 27.7–28.3 | 190–195 |
| mixed | 3.07 | 29.5% | 27.7–28.0 | 110 |
| prose | 1.43 | 6.1% | 27.4–27.8 | 52 |

- Without speculative decoding the step is 14.2 ms with the GPU busy 98.5% of it.
  Concurrency: spec 27.6 (1 request) → 28.9 (2) → 35.6 ms (4); no spec 14.2 → 16.8 ms (4).
- The 13.4 ms the spec step adds: GPU idle about 7.3 ms (the drafter's FlashInfer metadata
  build calls `seq_lens.to("cpu")`, an implicit device-to-host sync, and each of 15 KV
  cache groups rebuilt its metadata), drafter GPU 2.7 ms (1.41 ms of it a second read of
  the full bf16 lm_head), and verify-only kernel time about 3.5 ms.
- py-spy on the TP0 worker (one request): 67.3% of samples in the drafter's
  `seq_lens_cpu` read, 14.7% in GDN metadata builds for 10 groups, 6.9% in the rest of
  `execute_model`. The engine core spends 98.8% waiting for workers; the API server
  main thread is 99% idle.

nsys kernel time per step on GPU0 (kernel durations are reliable under tracing):

| Kernel | Target segment, ms | Drafter segment, ms | No spec, ms |
|---|---|---|---|
| Marlin W4A16 | 9.82 | 0.59 | 9.73 |
| custom all-reduce | 2.20 | 0.73 | 0.96 |
| `in_proj_a/b` bf16 GEMM | 1.67 | — | 0.27 |
| lm_head bf16 GEMM | 1.53 | 1.41 | 1.43 |
| GDN recurrence | 0.73 | — | 0.25 |
| elementwise | 0.57 | — | 0.08 |
| attention | 0.29 | 0.06 | 0.23 |
| logits all-gather | 0.17 | — | 0.04 |
| GPU busy, union | 17.6 | 2.7 | 14.07 |

- Marlin at M=8 costs the same as at M=1 (9.82 vs 9.73 ms, about 77% of memory
  bandwidth): verifying 8 tokens is close to free in the big GEMMs.
- `in_proj_a/b` (bf16, [48, 5120], left unquantized) at M=8 lands on a cutlass WMMA 16×16
  kernel with a grid of 4 CTAs, 34 µs per call; at M=1 a gemv + split-K pair takes 5.6 µs.
  E006 targets this; E002 counts it as +1,393.2 µs net per step.
- At 4 requests (32 verify tokens) the target segment is 25.1 ms busy, Marlin 14.7 ms of
  it, all-reduce 3.8, GDN 1.75, lm_head 1.45 ms; `in_proj_a/b` moves to a faster kernel
  (0.44 ms).
- nsys node tracing lengthened this spec step from 27.6 to 35.7 ms (`cudaGraphLaunch`
  about 3.6 ms per call) and the no-spec step only from 14.2 to 14.3 ms.
