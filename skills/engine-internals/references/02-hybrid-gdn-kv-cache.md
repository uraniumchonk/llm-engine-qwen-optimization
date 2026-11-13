# 02 Hybrid GDN KV cache

Scope: vLLM 0.33.0 hybrid KV cache manager (default uniform-page planner) with
Qwen3.8-27B (48 GDN + 16 attention layers) and the DFlash2 drafter (5 sliding-window
layers), fp8 KV, block size 1600, tensor parallel 2. Patch card: E005 in
[vllm/v0.33.0](../../../qwen3.8-27b/vllm/v0.33.0/README.md).

## Three kinds of layer state

| Layers | State | Size per layer per GPU at TP2 (arithmetic from the model config) |
|---|---|---|
| 16 target full attention | K and V per token: 4 KV heads (2 per GPU), head dim 256 | one 1600-token block in fp8: 1600 × 2 × 2 × 256 × 1 B = 1,638,400 B; per token for all 16 layers 16,384 B |
| 48 target GDN | a fixed recurrent state per request: conv state and SSM state | conv (10,240 / 2) × (4 − 1 + 7) × 2 B = 102,400 B; SSM (48 / 2) × 128 × 128 × 2 B (bf16) = 786,432 B; total 888,832 B |
| 5 drafter sliding window | K and V for a 2048-token window | same page as target attention |

- The conv state keeps the last 3 inputs plus one slot per speculative token (7), which is
  why its size depends on `num_speculative_tokens`.
- The planner gives every layer the same page size, so a GDN layer uses 888,832 of
  1,638,400 B (54%) of its page.
- Layers come in 16 repeats of (3 GDN + 1 attention), named `linear_attention` and
  `full_attention` in the model's layer list.

## How vLLM groups layers

The default uniform-page planner (`v1/core/kv_cache_planning.py:521` in 0.33; the same
rule sat in `v1/core/kv_cache_utils.py:1211` in 0.28):

1. Put layers with equal specs into buckets: GDN 48, attention 16, drafter 5.
2. Group size = the smallest bucket (5), unless the largest bucket is under 1.5 × the
   smallest (48 < 7.5 is false here).
3. Split each bucket round-robin (`layers[i::num_groups]`) into groups of that size. The
   coordinator creates one manager and block table per group, and every pool block has
   room for one page per slot of the widest group.

| Bucket | Group size 5 (upstream) | Group size 16 (E005) |
|---|---|---|
| GDN, 48 | 10 groups × 5 slots, 2 empty | 3 × 16, 0 empty |
| Attention, 16 | 4 × 5, 4 empty | 1 × 16, 0 empty |
| Drafter, 5 | 1 × 5, 0 empty | 1 × 16, 11 empty |
| Groups | 15 | 5 |

- The start log prints `Add 4 padding layers, may waste at most 25.00%` for attention at
  size 5. The percent is extra bytes over real layer bytes of that bucket, not the share
  of the pool.
- Pages per request, derived from the planner code (P = page bytes, N = attention blocks,
  M = resident GDN blocks per group, S = drafter blocks): size 5 uses
  P × (20N + 50M + 5S), size 16 uses P × (16N + 48M + 16S). Long contexts gain (N
  dominates); short requests can lose, because S pays 11 empty slots.
- E005 only allows padding in a bucket made entirely of drafter (`is_draft`)
  sliding-window layers; every target bucket must divide by the group size. With size 16
  the start log prints `E005: VLLM_KV_GROUP_SIZE overrides 5 -> 16 for layer counts
  [48, 16, 5]` and `Add 11 padding layers ... 220.00%` for the drafter.
- Fewer groups also means fewer per-group metadata builds. On 0.28, building GDN metadata
  for 10 groups took 14.7% of the worker's samples (about 4.1 ms per step). On 0.33 the V2
  runner reuses GDN metadata across groups with a per-group state-index regather, so the
  0.28 timing does not carry over.
- A new group size is a new KV layout. An external KV store needs a fresh directory, and
  the connector must map several groups: vLLM 0.33's bundled LMCache MP connector rejects
  more than one group (`lmcache_mp_connector.py:70`, `reformat_block_ids`); LMCache 0.5.6's
  own connector handles it ([04-lmcache-mp.md](04-lmcache-mp.md)).

## Align mode, speculative slots and block size

- With prefix caching on, 0.33 sets the mamba cache mode to `align` by itself (start
  log); `--mamba-cache-mode` has no effect in 0.33. LMCache 0.5.6's connector requires
  `align` and `max_num_batched_tokens >= block_size`.
- In align mode a request keeps 2 + `num_speculative_blocks` GDN blocks per group, plus one
  when the Qwen GDN layer exports prefill checkpoints (`v1/kv_cache_interface.py:1135`);
  with 7 speculative tokens that is 9 or 10 blocks. Estimate, not measured: on 0.28 with
  group size 5 the GDN slots of one request took about 737 MB per GPU at TP2
  (10 groups × 9 blocks × 5 slots × 1,638,400 B).
- Block size 1600 with a token budget of 2048 keeps `block ≤ budget < 2 × block`, the rule
  we use so that every block boundary has a GDN state for prefix caching. LMCache uses
  the same 1600-token chunk.
- E014 keeps `ceil(window / 1600)` = 2 drafter chunks up to each checkpoint, which covers
  the 2048-token window that ends at that chunk boundary.

## Fine-grained prefix hits and shared pages (E031)

- `--prefix-match-unit 64` hashes the prompt in 64-token units so a hit can end inside a
  1600-token block. 0.33 turns this off unless every prefix-cacheable group's manager
  supports fine-grained lookup; the DFlash drafter's `SlidingWindowManager` does not, and
  the start log prints `Disabling fine-grained prefix-cache hits ... SlidingWindowManager`.
  Hits then stay on the block grid: an agent turn recomputes 0–1,600 tokens depending on
  where the previous prompt ended (TTFT cycled about 200–900 ms on our rig).
- Without FlashInfer's GDN prefill checkpoints (SM90/SM10x/SM12x only; SM86 runs the
  Triton/FLA kernel) the scheduler splits the prefill at the prompt's last 64-token
  boundary (`tail_boundary`) and caches the GDN state there. That costs one extra forward
  step per prompt. With DFlash no EAGLE block is dropped, so the tail key and the GDN tail
  state sit at floor((P−1)/64)×64.
- A hit that ends mid-block shares a block the new request will write into, so vLLM
  copies it first (copy-on-write: `_apply_cow`, then `copy_kv_cache_blocks_inplace` in the
  worker).
- Pitfall we hit: `copy_kv_cache_blocks_inplace` skipped a per-layer tensor whose start
  address it had seen. With the E005 grouping, the GDN state view and the attention or
  drafter KV view of one layer slot share an allocation and start at the same address. The
  GDN view was copied through its own layout (a small part of each page) and the KV view
  was skipped, so the resumed request read KV that was partly zero (about 496K bytes per
  copied block). What it looked like: greedy output diverged within 4–17 tokens, KL
  against a full recompute was 13× the concurrency noise floor, an fp32 SSM cache did not
  change it, and block-aligned hits stayed bitwise exact. Fix in patch E031: de-duplicate
  by address, shape, stride and dtype. Whether stock grouping or other models hit the same
  path is not checked.
- Conversations that rewrite earlier messages fork before the previous prompt's tail,
  where no GDN state exists; they fall back to the block grid. The gain is for turns that
  extend or keep the previous prompt (fixed system prompt and tools: p50 796 → 230 ms).

## KV pool sizes we measured

From the start log line `GPU KV cache size: N tokens`; fp8 KV, `--max-model-len` 256k,
`--gpu-memory-utilization 0.86`, test instance without LMCache unless noted.

| Setup | KV pool, tokens | Run |
|---|---|---|
| 0.28.0 | 369,545 | E003 |
| 0.33.0 unpatched | 394,971 (available KV memory 8.66 GiB) | E004 |
| 0.33.0 + E003 | 396,364 (8.69 GiB) | E004 |
| 0.33.0 + E003 + E005 | 488,000 (8.96 GiB) | E005 |
| + PCIe IPC all-reduce | 486,666 | E007 |
| 0.33.0 without speculative decoding | 633,831 | E006 |
| 0.28.0, both attention backends Triton, bf16 KV, 128k context | 163,469 | EXP-1 |

- E005 adds 23.1% tokens while available KV memory grows 3.1%; tokens per GiB grow 19.4%.
- The no-speculation instance loads no drafter weights, has no drafter group and holds no
  speculative GDN slots; its pool is 633,831 tokens.
- Compare pools only with `--kv-cache-dtype`, `--max-model-len` and
  `--gpu-memory-utilization` written next to them.
