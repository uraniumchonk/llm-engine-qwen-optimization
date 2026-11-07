# Re-deriving the model-specific constants

Each patch carries one or two constants chosen for Qwen3.8-27B + DFlash2 at tensor
parallel 2. Re-derive them from your model and verify each with the check listed.

## E005: KV cache group size

1. List the buckets the planner forms: layers with equal KV spec. Ours: GDN 48,
   attention 16, drafter sliding window 5 (drafter layers are marked `is_draft`).
2. Candidates G: every common divisor of all target bucket counts. For 48 and 16:
   1, 2, 4, 8, 16. A bucket made only of drafter sliding-window layers may be smaller
   than G or not divide by it; E005 pads it with empty slots. A target bucket that does
   not divide makes the start fail with a ValueError.
3. Compare page use per request for each G (P = page bytes, N = attention blocks,
   M = resident GDN blocks per group, S = drafter blocks):
   `P × (attention groups × G × N + GDN groups × G × M + drafter groups × G × S)`.
   Ours: G = 5 gives P × (20N + 50M + 5S), G = 16 gives P × (16N + 48M + 16S). Long
   contexts are dominated by N, so the G with zero empty target slots and the fewest
   groups wins there; short requests pay the drafter's empty slots.
4. Set `VLLM_KV_GROUP_SIZE=G` and check: the start log prints `E005: VLLM_KV_GROUP_SIZE
   overrides <old> -> G for layer counts [...]`, the target padding warnings are gone,
   `GPU KV cache size` grows (ours 396,364 → 488,000 tokens), greedy output is identical
   to the run without the switch.
5. For an A/B baseline leave the variable unset. Setting it to the upstream value can
   fail (5 does not divide 48 or 16).

## E006: the small-M GEMM shape

- The layer: GDN `in_proj_ba`, a bf16 linear the AWQ checkpoint leaves unquantized. It
  projects the hidden state to the `b` and `a` gate inputs, one value each per GDN value
  head.
- N per GPU: 2 × value heads ÷ TP. Ours: 2 × 48 ÷ 2 = 48 (derived from our shape; check
  it against the weight's shape in your checkpoint).
- K: the hidden size. Ours 5120.
- M: tokens in the decode batch = requests × (speculative tokens + 1). Ours 8 at one
  request, 32 at four.
- Check on your GPU: an nsys trace of one decode step at your M. The patch pays when the
  kernel for this layer is far slower than its M=1 time; ours was a cutlass WMMA 16×16
  kernel with 4 CTAs at 34.7–38.1 µs per call against 5.6–5.87 µs at M=1.
- The Triton path runs for M ≤ 16 only. Above that it calls `F.linear`; at our M=32
  cuBLAS already picked a faster kernel (0.44 ms per step).
- After applying, confirm the kernel `_small_m_splitk_kernel` appears in the trace, and
  that `VLLM_E006_SMALL_M=0` brings the old kernel back. Compare decode logits with
  `logit_diff.py gen/gcmp` (ours: KL 4.11e-4 against a 1.72e-4 floor).
- The same check applies to any other unquantized small linear in your model; we only
  found this one.

## E014: checkpoint interval K and drafter window chunks

- Drafter chunks per checkpoint: `ceil(drafter window ÷ chunk size)`. Ours:
  ceil(2048 ÷ 1600) = 2 (rule in the E014 card).
- K trades store size against hits:
  - stored per K chunks: K attention objects, 1 GDN object, and the drafter chunks
    above (at most K);
  - estimate, not measured: bytes per chunk ≈ attention + (GDN + drafter chunks ×
    drafter) ÷ K; for ours at K=4: 26,214,400 + (43,008,000 + 2 × 8,192,000) ÷ 4 ≈
    41.1 MB per rank vs 77.4 MB without the patch;
  - measured at K=4 on our test workload: L2 bytes −43%; a 25,219-token production
    request stored about 1.26 GB vs 2.32 GB;
  - cost: a new session whose shared prefix ends off a checkpoint hits up to K−1 fewer
    chunks (ours 8,000 → 6,400 tokens).
- The K−1 cost applies only to prefixes shared across sessions that end off a
  checkpoint; the byte saving grows with K. Weigh both against your traffic.
- Check after setting `LMCACHE_MEOW_CKPT_K=K` in the vLLM process: the log prints the
  roles of your KV groups (ours `roles=['gdn', 'gdn', 'gdn', 'attention', 'drafter']`).
  Roles all `attention` mean detection failed and every chunk is stored. In L2, GDN
  objects should number about 1/K of attention objects plus the prompt-end checkpoints
  (ours 44 of 146 at K=4). Run kv_probe: zero digest mismatches.

## E003: when the software decode is needed

E003 has no tunable constant; decide whether you need it.

- Needed when: compute capability below 8.9, KV cache dtype `fp8` or `fp8_e4m3`, and a
  model part (our drafter) set to `TRITON_ATTN`. Without E003 the backend refuses at
  start: `FP8 KV cache is not supported by the Triton attention backend`.
- Check after applying: the start log prints `Triton attention: fp8 e4m3 KV via software
  dequant (E003)`, the drafter's graphs are captured `(FULL)`, the KV pool is unchanged,
  greedy output is identical to the FlashInfer drafter run.
- If the start fails with `type fp8e4nv not supported in this architecture`, some Triton
  kernel received an fp8 tensor, most often a quantized query; keep query quantization
  off.
- Measured gain for us: one request 27.10–27.47 → 21.77–22.10 ms per step on 0.33;
  at four requests 35.67–36.11 → 32.23–33.04 ms server ITL (E004).
