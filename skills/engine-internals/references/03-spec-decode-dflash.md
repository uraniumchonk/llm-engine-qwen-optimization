# 03 Speculative decoding with DFlash2

Scope: vLLM 0.33.0 (V2 model runner) with `method=dflash`, the DFlash2-W4A16 drafter
(5 sliding-window layers), 7 speculative tokens, target Qwen3.8-27B on FlashInfer,
drafter on `TRITON_ATTN` with E003. Earlier numbers from 0.28.0 are marked.

## What one step does

1. The target runs its 64 layers on 1 + 7 tokens per request (the last accepted token
   and the 7 drafts), then the lm_head on those 8 positions.
2. The rejection sampler keeps the longest prefix of drafts that matches the target (at
   temperature 0, the target's argmax) and adds one target token, so a step emits 1 to 8
   tokens per request.
3. The drafter proposes the next 7 tokens in one non-causal forward of its 5
   sliding-window layers over masked positions, conditioned on target hidden states
   (our start log lists auxiliary target layers 6, 20, 34, 48, 62). It scores candidates
   through the target's full bf16 lm_head (248,320 rows), a second full lm_head read per
   step. In 0.28 DFlash inherits the common speculative proposer and skips the
   sequential EAGLE loop; in 0.33 the V2 runner selects `DFlash2Speculator`.

- The step costs the same at any acceptance: on 0.28 code (acceptance 78.7%) and prose
  (6.1%) both ran 27.3–28.3 ms per step (EXP-0).
- The draft context length on the GPU is `target_seq_len − rejected + K + 1`. It depends on
  this step's rejections, which the CPU learns only after a device-to-host copy (E001).
- Drafter cost on 0.33 per step: 5 layers 1.37 ms (Marlin 0.77, bf16 GEMM 0.24,
  all-reduce 0.11) and the lm_head read 1.47 ms (E022). On 0.28 the drafter added 2.7 ms
  of GPU time (EXP-0).
- Speculation also costs KV pool: GDN speculative slots and the drafter's group (see
  [02-hybrid-gdn-kv-cache.md](02-hybrid-gdn-kv-cache.md)); 488,000 tokens with the drafter vs
  633,831 without (E005, E006).

## Acceptance and when speculation pays

Tokens per step at temperature 0, one request, 300 tokens (prose and mixed prompts ask
for Traditional Chinese):

| Prompt type | 0.28.0 production (EXP-0) | 0.33 production code (E017 base) |
|---|---|---|
| code | 6.51 (acceptance 78.7%) | 6.49 |
| numeric | 5.44 (63.5%) | not run |
| mixed | 3.07 (29.5%) | 2.80 |
| prose | 1.43 (6.1%) | 1.53 |

- Our production benchmark at temperature 0 (six prompt types, 1 and 4 requests)
  accepted 3,911 of 12,397 draft tokens, 0.315 (E016).
- Speculation pays when tokens per step exceed the ratio of spec to no-spec step time.
  Estimate, not measured: 0.28 needed more than 27.6 / 14.2 = 1.94 (E001); 0.33 with all
  patches needs more than 1.38–1.40 (19.75–19.93 ms over 14.23–14.31 ms, E017 base and
  E006).
- Measured on 0.33 with E003 + E005 + E006 (E006): one prose request 73.7 tok/s with
  speculation vs 70.4 tok/s without; four prose requests 168.5 vs 240.8 tok/s in total.
  At four requests the no-spec instance is faster on prose.

## The drafter's attention backend

| | FlashInfer (upstream default for the drafter here) | `TRITON_ATTN` + E003 |
|---|---|---|
| Metadata build | calls `seq_lens.cpu()`, a blocking device-to-host copy (0.33 `flashinfer.py:1669-1679`) | takes GPU `seq_lens` and block tables directly; no copy at this site |
| CUDA graph | non-causal fa2 on sm86 supports only single-token decode graphs, so the drafter runs eager; start log `FlashInferBackend does not support full CUDA graphs; running the draft eagerly` | full graph; start log `Capturing dflash2 CUDA graphs (FULL)` |
| fp8 KV on sm86 | supported | needs E003 (software e4m3 decode) |

- Measured: drafter moved to Triton with fp8 KV, step 27.10–27.47 → 21.77–22.10 ms on
  0.33 and 27.60–31.01 → 22.50–22.75 ms on 0.28, greedy 6/6 identical (E003, E004).
- On 0.28, py-spy put 67.3% of the TP0 worker's samples in the drafter's
  `seq_lens_cpu` read; nsys showed one long `cudaMemcpyAsync` per step with a 15.4 ms
  median (EXP-0). That time is mostly the CPU waiting for the target verify to finish.
- Rejected design (E001): feeding the FlashInfer builder a CPU upper bound of the
  lengths. It exposes rejected draft context, shifts causal and sliding-window
  boundaries, and can reference an extra page.
- The target still uses FlashInfer and its build still calls `seq_lens.cpu()`; on 0.33
  that wait is 16.4 ms per step at one request (E026). See
  [01-stack-top-down.md](01-stack-top-down.md).

## Changing the drafter or the verify width

- Validate a drafter-only change by greedy output identity against the baseline (k/N
  identical texts) and report acceptance separately. The target decides every emitted
  token, but a different verify batch shape can still flip a few tokens: in E012 one of
  22 requests differed at its third generated token with speculation on and none with it
  off, while kv_probe showed the restored KV bit-exact.
- A smaller candidate vocabulary for the drafter (E018, 32k rows) cut 1.1–1.3 ms per
  step but lowered code tokens per step 6.63 → 5.15. The rows were chosen from a corpus
  that did not match what the model writes; 17.0% of generated prose tokens were missing.
- Batch-size based dynamic K in 0.28 changes K for the whole batch; K=0 still runs the
  drafter forward before returning no drafts, and V1 full graph mode drops to piecewise
  (E001).
- Per-request low K (E021): verify width 1 for requests whose drafts keep failing raised
  4-request prose from 185 to 207–213 tok/s per request at KL 7.38e-4; K=0 made prose
  steps 25.3–79.7 ms. Open.
- 0.33 runs DFlash with async scheduling (no disable warning in our start log; upstream
  #58065). Upstream release notes before 0.33 also list in-graph DFlash draft metadata
  (#60692) and a fused DFlash2 grouped convolution (#55960).
