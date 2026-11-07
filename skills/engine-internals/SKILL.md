---
name: engine-internals
description: Load before you change, port or profile a patch in this repo. Index of the serving stack from the HTTP request down to kernels and the GPU link (vLLM 0.33 with LMCache 0.5.6, hybrid GDN model, DFlash2 drafter, RTX 3090 sm86, tensor parallel 2), with what we measured at each layer and which reference to read.
---

# Engine internals, top to bottom

Every number in these files was measured on the rig in
[qwen3.8-27b/README.md](../../qwen3.8-27b/README.md) unless it is labelled "estimate, not
measured". Experiment ids (E###) point to rows in
[qwen3.8-27b/results/README.md](../../qwen3.8-27b/results/README.md). Read the layer you
are about to touch, then the layers directly above and below it.

## Layers

1. **Request path and process model** ->
   [references/01-stack-top-down.md](references/01-stack-top-down.md).
   An HTTP request passes the API server, the engine core (scheduler, KV cache manager,
   KV connector) and one worker process per GPU, which replays CUDA graphs and samples.
   On our rig a speculative decode step is 19.3–19.9 ms of which the target model's 64
   layers take 14.2 ms of GPU time; the CPU critical path after the last host sync is
   816 µs (E022, E026). The scheduler and API server are not on the critical path at one
   request.

2. **Hybrid GDN KV cache** ->
   [references/02-hybrid-gdn-kv-cache.md](references/02-hybrid-gdn-kv-cache.md).
   48 Gated DeltaNet layers keep a fixed-size recurrent state per request; 16 attention
   layers keep KV per token; the drafter adds 5 sliding-window layers. vLLM groups layers
   of equal spec into KV cache groups and picks the group size from the smallest bucket,
   which E005 overrides (KV pool 396,364 → 488,000 tokens). Read before touching group
   size, block size, align mode or anything that changes the KV layout.

3. **Speculative decoding with DFlash2** ->
   [references/03-spec-decode-dflash.md](references/03-spec-decode-dflash.md).
   The target verifies 8 tokens per request per step; the drafter proposes 7 in one
   non-causal forward and reads the full lm_head again. Step time does not follow the
   acceptance rate, so prose (acceptance about 6%) gains little. Read before touching the
   drafter, its attention backend, the verify width or the rejection sampler.

4. **LMCache multiprocess server** ->
   [references/04-lmcache-mp.md](references/04-lmcache-mp.md).
   A separate server process holds KV and GDN state per 1600-token chunk in host RAM (L1)
   and on disk (L2), split into attention, GDN and drafter object groups. E014 stores GDN
   and drafter objects only at checkpoint chunks. Read before touching the connector,
   chunk size, object groups or anything that changes the KV layout.

5. **Kernels on Ampere sm86** ->
   [references/05-ampere-sm86.md](references/05-ampere-sm86.md).
   No fp8 tensor cores (E003 decodes fp8 KV in software for Triton), poor cuBLAS choices
   at small M (E006), Marlin W4A16 near memory bandwidth at M=8, a 250 W power cap that
   sets the clocks. Read before writing or swapping a kernel.

6. **torch.compile and CUDA graphs** ->
   [references/06-torch-compile-cuda-graphs.md](references/06-torch-compile-cuda-graphs.md).
   Python branches in model code are fixed at trace time; run-time dispatch needs a
   custom op (E006c). The 0.33 compile cache keys on source, not on environment
   variables. Profilers inflate graph launch time. Read before adding any shape- or
   switch-dependent code path to the model.

7. **Multi-GPU communication** ->
   [references/07-multi-gpu-communication.md](references/07-multi-gpu-communication.md).
   139 all-reduces per step over PCIe gen4 x16 between two GPUs; custom all-reduce 8.19 µs
   vs FlashInfer PCIe IPC 3.30 µs per 80 KB call (E022). Read before changing the
   tensor-parallel size, the all-reduce backend or the GPU topology.

## Related

- Step breakdown with all tables: [step-anatomy.md](../../qwen3.8-27b/results/step-anatomy.md).
- How to measure each layer without fooling yourself: [measurement](../measurement/SKILL.md).
- How these facts map onto another GPU or model: [port-to-your-hardware](../port-to-your-hardware/SKILL.md).
