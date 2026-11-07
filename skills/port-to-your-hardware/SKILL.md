---
name: port-to-your-hardware
description: Load when you want the approach of this repo on your own GPUs, model, drafter or quantization instead of our exact rig. Inventory to take first (GPU arch and sm, memory, interconnect and P2P, CPU, layer types and counts, drafter, quantization), a per-patch table of which conditions make each patch transfer as is, need a re-derived constant, or not apply, how to re-derive E005, E006 and E014 constants, and where to start measuring top down.
---

# Port to your hardware

The patches in [qwen3.8-27b](../../qwen3.8-27b/README.md) were made for one model on one
rig, and we test nothing else. What transfers is the method and the conditions each
patch depends on. Take the inventory, decide per patch with the table, re-derive the
constants ([references/derive-constants.md](references/derive-constants.md)), then
measure your own baseline before you apply anything. Our numbers are a reference for
the shape of the problem, not a prediction for your setup.

## 1. Inventory first

Write each item down with the command output before the first experiment.

| Item | How to read it | Ours |
|---|---|---|
| GPU, arch, sm | `nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit --format=csv` | 2× RTX 3090, 8.6, 24 GB class, 250 W |
| Clocks under load | `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active --format=csv -lms 200` during decode | SM 1,545–1,575 MHz, memory 9,501 MHz, software power cap active |
| Interconnect and P2P | `nvidia-smi topo -m`, `nvidia-smi topo -p2p r` | PXB (one PCIe switch), gen4 x16, P2P OK |
| CPU | `lscpu`; governor in `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | 6 cores / 12 threads, `performance` |
| Model layer types and counts | model `config.json` (layer type list, hidden size, head counts and dims) or the KV planner's start log | 48 GDN (`linear_attention`) + 16 `full_attention`, hidden 5120, GDN 48 value heads / 16 key heads of dim 128, attention 4 KV heads of dim 256, vocabulary 248,320 |
| Drafter | speculative config: method, layers, attention type and window, tokens | DFlash2, 5 sliding-window layers, window 2048, 7 tokens |
| Quantization | checkpoint `quantization_config`: method, bits, group, symmetry, ignore list | AWQ-INT4 (compressed-tensors, asymmetric uint4 with zero points, group 32); `in_proj_ba`, embeddings, lm_head and the GDN output projection of layer 0 stay bf16 |
| Engine and kernels | `pip show vllm lmcache torch triton flashinfer-python`; start log lines for attention backend, all-reduce backends, linear kernel, CUDA graph mode | vLLM 0.33.0, LMCache 0.5.6, torch 2.14.0+cu132, Triton 3.8.0, FlashInfer 0.7.2.post1; Marlin |
| KV layout | start log: `GPU KV cache size`, `Add N padding layers`, the KV cache group count | 394,971 tokens and 15 groups before E005 |

Also measure, before any patch: the spec and no-spec step time, tokens per step per
prompt type, the KV pool ([measurement](../measurement/SKILL.md)).

## 2. Patch by patch

| Patch | Condition on your setup | Verdict |
|---|---|---|
| E003 Triton fp8 KV, software decode | GPU below SM89, fp8 e4m3 KV, and a component (our drafter) that you want on `TRITON_ATTN` | transfers as is; the code path is off on SM89+ and on non-CUDA platforms |
| | SM89 or newer | does not apply: Triton has native fp8 there |
| | e5m2 KV or bf16 KV | does not apply (the bit trick is e4m3fn only) |
| | your drafter on FlashInfer has no blocking `seq_lens.cpu()` in its build and gets a full CUDA graph | measure first; the reason for E003 may be absent (on sm86 the FlashInfer TRTLLM path is unavailable; other archs differ and we did not measure them) |
| E005 KV group size | hybrid model whose smallest bucket is a small drafter sliding-window bucket and the start log prints `Add N padding layers` for target buckets | re-derive the group size |
| | no drafter, or the planner's group size already divides every target bucket | does not apply |
| | an external KV store (LMCache) | a new group size needs a fresh store directory and a connector that maps several groups |
| E006a/c small-M GEMM | a bf16 linear left unquantized whose shape gets a poor cuBLAS kernel at your spec-decode M (check nsys) and M ≤ 16 | re-derive: confirm the shape, M and the kernel cuBLAS picks; the Triton path runs only for bf16, 2-D input, no bias, M ≤ 16 |
| | M above 16 at your batch (requests × (speculative tokens + 1)) | does not apply at that batch; widen the bound only after a kernel test |
| E006b GDN views | vLLM's Qwen GDN layer with speculative decoding | transfers as is; measured only together with E006a/c |
| E014 checkpoint store | hybrid recurrent model, LMCache multiprocess server with `--separate-object-groups`, GDN groups plus one attention group and one drafter sliding-window group (what we tested) | re-derive K and the drafter window chunks; check the roles log line; other group mixes are untested |
| | server without `--separate-object-groups` | do not enable: a null block id is read as block 0, which is wrong data |
| | LMCache blend mode | does not apply on 0.5.6 with separate groups for this kind of model |
| PCIe IPC all-reduce (upstream switch) | tensor parallel 2 over PCIe with P2P | try it: −0.3 to −0.6 ms per step for us, KL 4.64e-4 |
| | tensor parallel above 2 | we have not checked whether it runs; the kernel we saw is TP2-specific |
| Drafter on `TRITON_ATTN` (flag, needs E003 on sm86 with fp8) | the drafter's attention backend syncs or runs eager (start log) | try it; it was our largest single gain (about 5 ms per step) |

Card details for every patch, including the "Keep when porting" lines:
[vllm/v0.33.0](../../qwen3.8-27b/vllm/v0.33.0/README.md),
[vllm/v0.28.0](../../qwen3.8-27b/vllm/v0.28.0/README.md),
[lmcache/v0.5.6](../../qwen3.8-27b/lmcache/v0.5.6/README.md).

## 3. Where to start, top down

1. Baseline on your rig: spec step, no-spec step, tokens per step per prompt type, KV
   pool, GPU utilization during decode. The spec minus no-spec gap is what speculation
   costs you; ours was 13.4 ms on 0.28 and 5.0–5.7 ms on 0.33 with patches.
2. Read the start log for the cheap wins first: drafter attention backend and CUDA graph
   mode (`running the draft eagerly`), KV padding warnings, all-reduce backends.
3. nsys census of one spec step against one no-spec step at the same request count;
   rank kernel classes by the difference ([engine-internals](../engine-internals/SKILL.md)).
   Look first at: GPU idle before the target graph, the drafter's second lm_head read,
   small-M bf16 GEMMs, all-reduce.
4. CPU side without a profiler if GPU idle remains; profilers inflate exactly that part.
5. Then capacity (group size, speculative slots), then the KV store (LMCache object
   groups, checkpoints).
6. Per change: the loop in [optimization-method](../optimization-method/SKILL.md),
   validation by class in [measurement](../measurement/SKILL.md).

What does not transfer: our step times, KV pools, KL values (measure your own noise
floor), LMCache L2 directories (layout-specific), compile caches.

Not tried on our rig: unlocking the GeForce half-rate tensor mode, power limits above
250 W, locked memory clocks.
