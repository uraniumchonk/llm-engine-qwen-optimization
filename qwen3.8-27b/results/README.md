# Qwen3.8-27B results reference

Every experiment we ran for this model, E001–E026 plus the two early runs EXP-0 and
EXP-1, with its outcome and the numbers that decided it. Patch details live in the cards:
[vLLM 0.33.0](../vllm/v0.33.0/README.md), [vLLM 0.28.0](../vllm/v0.28.0/README.md),
[LMCache 0.5.6](../lmcache/v0.5.6/README.md). The decode step breakdown is in
[step-anatomy.md](step-anatomy.md). How the numbers are taken:
[measurement skill](../../skills/measurement/SKILL.md). Whole-stack benchmark tables and
charts (all prompt types, concurrency levels, engine versions) are in the notes repo
[vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes); this
page keeps one outcome and the deciding numbers per experiment.

Default setup for every row unless the row says otherwise:

- Rig and flags of the [model README](../README.md#test-rig): Qwen3.8-27B AWQ-INT4 with
  the DFlash2 drafter (7 speculative tokens), tensor parallel 2 on 2× RTX 3090, fp8 KV.
- A test instance without LMCache, one request, greedy (temperature 0), 300 output tokens,
  2 repetitions per prompt type (code, prose, mixed). Our prose and mixed prompts ask for
  Traditional Chinese output, which is why prose acceptance is low.
- "Step" is decode wall time divided by the number of speculative verify steps
  (rule 1 in the measurement skill). Ranges are min–max over repetitions and prompt
  types. "4 requests" rows divide by per-request steps, so they are per request-step.
- KL is the decode-logit KL over the top-20 tokens against the unpatched or previous
  build; the same-build floor with one concurrent request is 1.72e-4 (E011).

Outcome words: adopted (in our production), dead (measured and dropped), blocked (could
not run), open (measured, decision pending), measured (an analysis run, no switch to
adopt).

## Experiments

| Id | Date | Question | Outcome | Key numbers |
|---|---|---|---|---|
| EXP-0 | 2026-11-03 | Where does a speculative step go on our vLLM 0.28.0 production instance? | measured | Spec step 27.3–28.3 ms for all four prompt types (code acceptance 78.7%, prose 6.1%), so step time does not follow acceptance. No-spec step 14.2 ms, GPU busy 98.5%. The extra 13.4 ms: GPU idle about 7.3 ms, drafter 2.7 ms, verify-only kernels about 3.5 ms. |
| EXP-1 | 2026-11-03 | Does moving target and drafter attention to `TRITON_ATTN` remove the drafter's host sync? | dead (capacity), led to E003 | Step 27.3–27.7 → 20.9–21.2 ms. Needed bf16 KV and 128k context: KV pool 369,545 → 163,469 tokens (two settings changed at once). |
| E001 | 2026-11-03 | Which design removes the drafter's sequence-length readback on 0.28 and keeps fp8 KV? | adopted as E003 | Chosen: drafter-only Triton with software fp8 decode. Rejected: a CPU upper bound for the lengths (they depend on this step's rejections). Hardening: bytes 0x7F/0xFF decode to NaN, fp8 query rejected; 5 CPU tests over all 256 encodings. |
| E002 | 2026-11-03 | Which kernels make the 0.28 spec step longer than a no-spec step (nsys)? | measured | Kernel time 21,745.0 vs 14,094.3 µs per step (+7,650.7 µs; 136 / 149 steps). Largest: drafter's second full lm_head +1,472.9 µs net, custom all-reduce +1,990.4 µs, `in_proj_ba` WMMA at M=8 +1,393.2 µs net (34.7 µs per call × 48). |
| E003 | 2026-11-03 | Can the drafter run Triton attention on fp8 KV on sm86? | adopted | 0.28: step 27.60–31.01 → 22.50–22.75 ms (first JIT repetition 35.12 ms left out), KV pool 369,545 tokens in both, greedy 6/6 identical. 0.33: 27.10–27.47 → 21.77–22.10 ms. |
| E004 | 2026-11-03 | Does vLLM 0.33.0 keep the 0.28 problems, and do E003 and a group size switch port? | adopted (0.33.0 base) | Unpatched 0.33.0: step 27.10–27.47 ms, KV pool 394,971 tokens, FlashInfer drafter still eager with a `seq_lens.cpu()` call. Greedy text 2/6 identical to 0.28 (target numerics changed). |
| E005 | 2026-11-03 | Does a forced hybrid KV group size of 16 recover the padded slots? | adopted | KV pool 396,364 → 488,000 tokens (+23.1%); available KV memory 8.69 → 8.96 GiB; step 21.77–22.10 → 21.03–21.30 ms; greedy 6/6 identical. |
| E006 | 2026-11-03 | Do a small-M GEMM for `in_proj_ba` and view-based GDN spec inputs help on 0.33? | adopted after E011 | First run 20.84–21.02 ms not counted (log overwritten, compile state unknown); E008 then showed the small-M kernel never ran. No-spec reference on 0.33: 14.23–14.31 ms, KV pool 633,831 tokens; 4-request prose 240.8 tok/s no-spec vs 168.5 tok/s spec. |
| E007 | 2026-11-03 | Does FlashInfer PCIe IPC all-reduce shorten the TP2 step? | adopted later (E017, E025) | Step 21.03–21.30 → 20.47–20.57 ms; greedy text 0/6 identical, so we waited for a KL check; KV pool 486,666 tokens. |
| E008 | 2026-11-03 | What separates the 0.33 patched spec step from no-spec (nsys)? | measured, led to E011 | Kernel gap 5,997.5 µs per step (128 / 199 steps). `in_proj_ba` still on cutlass WMMA: 1,818.0 µs per step, 38.1 µs per call; the E006a kernel was absent. Drafter lm_head 1,463.6 µs per step. |
| E009 | 2026-11-04 | Full-state backup of the 0.28 stack before migrating | done (operations) | Restore script with dry-run and apply modes; the dry-run reported 0 drifted files. |
| E010 | 2026-11-04 | What does LMCache 0.5.6 store for this model, and can GDN and drafter state be stored at checkpoints only? | design adopted as E014 | Per chunk per rank 77,414,400 B = GDN 43,008,000 + attention 26,214,400 + drafter 8,192,000 B. 0.5.6 already stores only the real GDN state. Blend mode and `--separate-object-groups` cannot be combined on this model. |
| E011 | 2026-11-04 | Why did E006a not run, and does a custom op fix it? | adopted (E006c) | The M ≤ 16 branch was fixed at torch.compile trace time. With E006c: step 21.15–21.40 (off) → 19.74–19.91 ms (on); KL 4.11e-4 over 969 tokens vs floor 1.72e-4 over 1,363; prefill logits bitwise identical at 6,639 positions. |
| E012 | 2026-11-04 | Is LMCache 0.5.6 retrieve correct on 0.33 with the patches (blend vs default engine with separate object groups)? | adopted (default + separate) | 22/22 requests hit from L1 and from L2 after a server restart. Outputs equal cold compute 21/22 with spec on, 22/22 with spec off. kv_probe: 936 and 780 digests, 0 differ. L2 prefetch of a 71,899-token prompt: 1,132.6 ms blend, 409.8 ms separate. |
| E013 | 2026-11-04 | Cut production over to 0.33 + E003/E005/E006 + LMCache 0.5.6 | adopted | Rehearsal with production settings and production sampling: step 20.09–20.25 ms, code 261.5–284.9 tok/s. A 20,319-token request: 11.42 s cold, 1.07 s on repeat (19,200 cached tokens). |
| E014 | 2026-11-04 | Store GDN and drafter state only at checkpoint chunks (K=4)? | adopted | L2 objects attention 146, GDN 44, drafter 82 (146 each without the patch); 6.39 vs 11.30 GB (−43%). Second-pass cached tokens 3,200 / 8,000 / 22,400 / 70,400, as predicted. kv_probe 932 digests, 0 differ. Cost: a cross-session first hit 8,000 → 6,400 tokens. |
| E015 | 2026-11-04 | Deploy E014 with K=4 in production | adopted | A 25,219-token request (15 chunks) stored attention 30, GDN 8, drafter 16 objects (both ranks): about 1.26 GB, 2.32 GB without checkpointing. |
| E016 | 2026-11-04 | Production speed vs our 2026-09-11 record on vLLM 0.28.0 | measured | Single-stream decode, old 3 runs → new 1 run: code 259.6–268.7 → 374.5 tok/s, agent job 142.9–167.6 → 224.0 tok/s; ITL p50 19.4–20.3 ms. The old record used other memory and batch settings. Full tables: notes repo linked above. |
| E017 | 2026-11-04 | Do the Triton GDN decode kernel or PCIe IPC all-reduce shorten the production step? | PCIe IPC adopted (E025); Triton GDN dead (E023) | Base 19.75–19.93 ms; `VLLM_GDN_DECODE_KERNEL=triton` 19.36–19.52 ms (KL 4.54e-4); PCIe IPC 19.32–19.52 ms (KL 4.64e-4). Same-process rerun of base: KL 0. |
| E018 | 2026-11-04 | Drafter candidates from a vocabulary subset (32k or 64k rows) instead of the full lm_head? | open (this subset dead) | 32k: step 19.70–19.86 → 18.50–18.69 ms, but code tokens per step 6.63 → 5.15 and code 334 → 277 tok/s; prose 72 → 83 tok/s. KV pool 486,666 → 470,666 tokens. 17.0% of generated prose tokens fall outside both subsets. |
| E019 | 2026-11-04 | A second Triton GDN spec kernel ("triton2")? | dead | Step 19.44–20.08 ms vs 19.35–19.52 ms for the existing Triton kernel; KL 4.70e-4. |
| E020 | 2026-11-04 | What runs on the CPU while GPU0 idles in the prep phase (nsys, torch profiler)? | measured | Prep idle median 1,611.3 µs under nsys, 954.1 µs of it inside `cudaGraphLaunch`. Estimate, not measured: unprofiled idle 0.8–0.95 ms (unprofiled step 19.75–19.89 ms minus 18.94 ms of kernel-busy time). |
| E021 | 2026-11-04 | Lower the verify width for requests whose drafts keep failing? | open | Low K=1: 4-request prose 185 → 207–213 tok/s per request; one request within 0.4 ms of off. KL 7.38e-4 and 7.77e-4. Low K=0 dead: prose steps 25.3–79.7 ms. |
| E022 | 2026-11-04 | Decode step anatomy of production (nsys) | measured | GPU cycle median 20.61 ms = prep 1.82 + target layers 14.22 + target lm_head 1.48 + verify 0.21 + drafter layers 1.37 + drafter lm_head 1.47 ms (126 cycles). 139 all-reduces per step. See [step-anatomy.md](step-anatomy.md). |
| E023 | 2026-11-04 | Both E017 switches together; other int4 GEMM kernels | dead | Both: 19.25–19.51 ms, no gain over one switch (19.32–19.52). TritonW4A16 47.11–47.66 ms. Humming 19.83–20.18 ms. |
| E024 | 2026-11-04 | Marlin with int8 activations (W4A8-INT8)? | blocked | Engine start fails: Marlin with 8-bit activations needs symmetric `uint4b8` weights; our checkpoint is asymmetric uint4 with zero points. |
| E025 | 2026-11-04 | Step time under the CPU governor `performance`; PCIe IPC final check | adopted (PCIe IPC in production since 2026-11-04) | Base 19.75–19.93 → 19.58–19.71 ms; PCIe IPC 19.28–19.42 ms. 4-request code 8.87–8.94 ms per request-step over 3 runs (`schedutil` single runs that day 8.99–13.08). Same tree + same compile cache across a restart: logits bitwise identical. |
| E031 | 2026-11-13 | Prefix hits on a 64-token grid instead of the 1600-token block (agent-loop TTFT) | adopted (E031) | Drafter sliding-window fine-grained lookup + copy-on-write fix for shared hybrid pages (before the fix 64-grid hits read partly zeroed KV: KL vs recompute 1.2e-2, noise floor 9.4e-4; after: 6.7e-4, exact resume bitwise identical). Replay of 4 stored agent chains, max_tokens 1: fixed system prompt + tools, 29.5K tokens, TTFT p50 796 / 805 → 230 / 229 ms, p90 1,247 / 948 → 538 / 497 ms; chains that rewrite earlier messages and 2K-token chains unchanged; drafter acceptance 0.242 vs 0.241; KV pool 488,000 unchanged |
| E026 | 2026-11-04 | CPU timestamps per step without a profiler | measured | Worker waits in `seq_lens.cpu()` 16.4 ms (1 request) / 27.1 ms (4). After the wait to the end of graph launch: 816 / 1,036 µs. Graph launch call 58 µs. Engine step with 4 requests 30.5 ms. |

## Adopted path

Single-request greedy step on the default setup, in the order we stacked the changes:

| Stack | Step, ms | KV pool, tokens | Source |
|---|---|---|---|
| 0.28.0 control (FlashInfer drafter) | 27.60–31.01 | 369,545 | E003 |
| 0.28.0 + E003 | 22.50–22.75 | 369,545 | E003 |
| 0.33.0 unpatched | 27.10–27.47 | 394,971 | E004 |
| + E003 | 21.77–22.10 | 396,364 | E004 |
| + E005 (group size 16) | 21.03–21.30 | 488,000 | E005 |
| + E006a/b/c | 19.74–19.91 | 488,000 | E011 |
| + PCIe IPC all-reduce, CPU governor `performance` | 19.28–19.42 | not read | E025 |
| 0.33.0 without speculative decoding, for reference | 14.23–14.31 | 633,831 | E006 |

Decisions and the measured reason for each:

- 0.33.0 is the base (chosen 2026-11-03): E003 and E005 port to it with greedy output
  unchanged against unpatched 0.33.0 (E004).
- The drafter runs `TRITON_ATTN` on fp8 KV through E003; the target stays on FlashInfer
  with fp8 KV, so the KV pool keeps its size (E003, EXP-1).
- `VLLM_KV_GROUP_SIZE=16`: +23.1% KV pool with greedy output unchanged (E005).
- E006a/b/c: −1.3 to −1.6 ms per step per prompt type at KL 4.11e-4 (E011).
- PCIe IPC all-reduce: −0.3 to −0.6 ms per step per prompt type at KL 4.64e-4 (E017,
  E025).
- LMCache 0.5.6 default engine with `--separate-object-groups`: retrieve verified bitwise
  and L2 reload of a 71,899-token prompt 1,132.6 → 409.8 ms (E012).
- E014 with K=4: L2 bytes −43% on our test workload, retrieve verified bitwise (E014).

Production speed after the cutover is in E016. The flags of the running stack are in the
[model README](../README.md#test-rig); the switches are in its patch table.

## Dead ends

Each was measured on the default setup unless noted; the numbers are in the table above.

- Both attention backends on `TRITON_ATTN` (EXP-1): fastest step of its day, but Triton on
  sm86 then needed bf16 KV, and the pool fell to 163,469 tokens. E003 keeps fp8 instead.
- Drafter vocabulary subset (E018, 32k rows): the step got 1.1–1.3 ms shorter, code tokens per step
  fell 6.63 → 5.15, so code tok/s fell 334 → 277. The subset was chosen from a file corpus
  that held mostly code, English and JSON; a corpus-balanced CPU-only re-selection still
  left 33.4% of held-out prose tokens outside a 32,768-row subset.
- Second Triton GDN spec kernel "triton2" (E019): slower than the existing Triton kernel;
  we took no nsys trace of it.
- Triton GDN decode kernel together with PCIe IPC (E023): the two do not stack.
- Other int4 linear kernels on sm86 (E023): TritonW4A16 makes the step 2.4× longer;
  Humming is 0–0.3 ms slower than Marlin (that run overlapped other CPU load). Exllama
  refuses bf16 activations.
- Marlin W4A8-INT8 (E024): blocked by our checkpoint's asymmetric int4 format.
- No speculation for failing requests (E021, low K=0): the drafter still runs, and prose
  steps grow to 25.3–79.7 ms.
- Lower GPU power limit (EXP-0 day, vLLM 0.28.0, one request): 250 → 225 W kept the step
  (code 29.0 → 27.8 ms, prose 27.9 → 27.9 ms); 200 W made it up to 7% longer (prose
  29.8 ms) while the SM clock fell 16% (1,823 → 1,530 MHz).

Not tried: unlocking the GeForce half-rate tensor mode, a power limit above 250 W, and
locked memory clocks.

## Open

- Drafter vocabulary subset (E018): the next attempt selects the subset from text the
  model itself generates.
- Low verify width for failing requests (E021, K=1): 4-request code ran one repetition per
  setting under `schedutil`, and those runs spread 8.99–13.08 ms per request-step on the
  same code, so the code side needs repeated runs under `performance` before a decision.
- CPU critical path (E026): the worker waits 16.4 ms in the target's FlashInfer
  `seq_lens.cpu()` and then spends 816 µs before the graph launch ends. A target attention
  backend that needs no CPU sequence lengths is the candidate; it has not been measured,
  and its A/B needs long-context prompts.
- Four GPUs: vLLM 0.33 disables its custom all-reduce on more than two PCIe-only GPUs and
  falls back to NCCL; we have not measured tensor parallel 4 on 0.33.
