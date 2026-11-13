# vLLM 0.33.0

Upstream: `vllm-project/vllm` tag `v0.33.0`. Apply order: [`series`](series).
In our production since 2026-11-04 together with [LMCache 0.5.6 + E014](../../lmcache/v0.5.6/README.md).
Rig and flags: [model README](../../README.md#test-rig).
Scope: written for Qwen3.8-27B (48 GDN + 16 attention layers) with the DFlash2 drafter on
2× RTX 3090 only; no attempt at generality. See [Scope](../../README.md#scope-one-model-one-rig).

| Patch | Switch | Files |
|---|---|---|
| E003 | `VLLM_TRITON_FP8_SW_DEQUANT=1` | `v1/attention/backends/triton_attn.py`, `v1/attention/ops/triton_unified_attention.py` |
| E005 | `VLLM_KV_GROUP_SIZE=<int>` | `v1/core/kv_cache_planning.py` |
| E006a | on when applied (`VLLM_E006_SMALL_M=0` off, with E006c) | new `model_executor/layers/mamba/gdn/small_m_bf16_gemm.py`, `model_executor/layers/linear.py`, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` |
| E006b | on when applied | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `third_party/flash_linear_attention/ops/fused_sigmoid_gating.py` |
| E006c | `VLLM_E006_SMALL_M=0` turns E006a/c off | `model_executor/layers/mamba/gdn/small_m_bf16_gemm.py`, `linear.py`, `qwen_gdn_linear_attn.py` |

Paths are under `vllm/`.

Single-request decode step time, greedy, 300 tokens, LMCache off (2026-11-03/04):

| Setup | ms per step | KV pool tokens |
|---|---|---|
| 0.33.0 unpatched | 27.10–27.47 | 394,971 |
| + E003 | 21.77–22.10 | 396,364 |
| + E003 + E005 | 21.0–21.3 | 488,000 |
| + E003 + E005 + E006a/b/c with `VLLM_E006_SMALL_M=0` | 21.15–21.4 | 488,000 |
| + E003 + E005 + E006a/b/c | 19.74–19.91 | 488,000 |
| 0.33.0 without speculative decoding, for reference | 14.23–14.31 | 633,831 |

Each speculative step verifies up to 7 drafted tokens, so tokens per second depends on the
acceptance rate. Tokens per second by prompt type, before and after:
[vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes).
Every experiment behind these rows: [results](../../results/README.md).

Correction 2026-11-07: the unpatched and + E003 rows first read 27.1–27.9 and 21.8–22.3 ms;
those ranges also held production-sampling runs. The greedy runs are above (run record E004).

---

### E003 Triton fp8 e4m3 KV on pre-SM89 GPUs

- Switch: `VLLM_TRITON_FP8_SW_DEQUANT=1`. Off by default; when off, or on SM89 and newer,
  or on non-CUDA platforms, the code path is upstream's.
- Files: `vllm/v1/attention/backends/triton_attn.py` (module flag `_SW_FP8_KV`, the fp8
  check in `__init__`, `supports_quant_query_input`, the KV view in `forward`,
  `do_kv_cache_update`); `vllm/v1/attention/ops/triton_unified_attention.py`
  (`_decode_e4m3fn`, `_cast_kv_tile`).
- Problem: the Triton attention backend refuses fp8 KV below SM89 (`native FP8 (fp8e4nv)
  requires SM89+`); Triton cannot compile fp8e4nv loads and stores on sm86 (we got a
  `CompilationError` for both with Triton 3.7.1). FlashInfer serves fp8 KV on sm86, but
  its metadata build for the DFlash drafter calls `seq_lens.to("cpu")` every step
  (`v1/attention/backends/flashinfer.py:1670-1679` in 0.33.0), a host sync that runs eager.
- Change: KV writes go through the CUDA op `reshape_and_cache_flash`, the same op the
  FlashInfer backend uses on sm86, with the cache viewed as uint8. Reads pass the cache to
  Triton as uint8; `_cast_kv_tile` decodes e4m3fn bytes with bit operations into fp16,
  multiplies by 2^8 to fix the exponent bias, then applies the tensor scale. The query
  stays bf16. The target keeps FlashInfer + fp8; the drafter moves to Triton + fp8.
- Keep when porting: bytes 0x7F and 0xFF decode to NaN (not ±480). Only `fp8` and
  `fp8_e4m3` are accepted (the bit trick is wrong for e5m2). Writes always use the CUDA op;
  the Triton write kernel is not touched. Query quantization is off while the switch is on
  (`tl.static_assert` rejects an fp8 query). `_SW_FP8_KV` is true only below SM89.
- Signs it is off or broken: `FP8 KV cache is not supported by the Triton attention
  backend` at start (switch unset); `type fp8e4nv not supported in this architecture` (some
  Triton kernel received an fp8 tensor, e.g. a quantized query). The log line
  `Triton attention: fp8 e4m3 KV via software dequant (E003)` confirms it is on.
- Measured: kernel test on 2026-11-03: written bytes equal torch's fp8 cast bit for bit;
  attention output vs a bf16 reference differs by at most 2.4e-4 (causal) and 7.8e-3
  (non-causal), from tile size 32 vs 16 accumulation order; all 256 encodings decode equal
  to independent arithmetic. GPU A/B on 0.33.0, greedy: decode step 27.10–27.47 → 21.77–22.10 ms,
  greedy outputs 6/6 identical.
- Written by: Claude (0.28.0 version), hardened by Codex; ported to 0.33.0 by Codex.
- Upstream status (2026-11-07): we found no upstream PR doing this.

### E005 Hybrid KV cache group size override

- Switch: `VLLM_KV_GROUP_SIZE=<positive int>` (we use 16). Unset = upstream.
- Files: `vllm/v1/core/kv_cache_planning.py` (the default uniform-page grouping function).
- Problem: the hybrid KV cache manager sizes groups by the smallest layer count among the
  attention types. With the 5-layer DFlash2 drafter that is 5: the 16 target attention
  layers become 4 groups × 5 slots with 4 empty slots, and the 48 GDN layers 10 groups.
  0.33.0 logs `Add 4 padding layers, may waste at most 25.00%`.
- Change: the switch sets the group size. With 16: attention 1 group × 16, GDN 3 groups,
  and the drafter's group padded from 5 to 16 slots.
- Keep when porting: only a bucket made entirely of drafter (`is_draft`) sliding-window
  layers may be padded with empty slots. Every target bucket must divide exactly, or start
  fails with a ValueError; there is no silent fallback. A new group size changes the KV
  layout, so an external KV store (LMCache L2) needs a fresh directory, and the connector
  must handle several KV groups (the built-in `lmcache_mp_connector.py:70`
  `reformat_block_ids` in 0.33.0 does not).
- Signs it is off or broken: no `E005: VLLM_KV_GROUP_SIZE overrides 5 -> 16` line at start
  (other grouping paths return before this code); `VLLM_KV_GROUP_SIZE must be a positive
  integer`; a divisibility ValueError. Setting 5 also fails (48 and 16 are not multiples of
  5), so an A/B baseline leaves the variable unset.
- Measured (2026-11-03, E003 on in both): KV pool 396,364 → 488,000 tokens (+23.1%),
  decode step 21.77–22.10 → 21.03–21.30 ms, greedy outputs 6/6 identical.
- Written by: Codex.
- Upstream status (2026-11-07): the vLLM hybrid KV cache manager design doc says the
  grouping heuristic will be revisited when a model gives a bad result; we found no
  override upstream.

### E006a Small-M split-K Triton GEMM for GDN `in_proj_ba`

- Switch: on when applied; `VLLM_E006_SMALL_M=0` turns it off (read inside the E006c op).
- Files: `vllm/model_executor/layers/linear.py` (`UnquantizedLinearMethod.apply` calls the
  Triton path for layers tagged `small_m_bf16_gemm`), `qwen_gdn_linear_attn.py` (tags
  `in_proj_ba`), new `vllm/model_executor/layers/mamba/gdn/small_m_bf16_gemm.py` (the kernel).
- Problem: during spec decode `in_proj_ba` runs at M=8, N=48, K=5120 and lands on a cutlass
  16×16 WMMA bf16 kernel: 34.7 µs per call in isolation, 38.1 µs per call and 1,818 µs per
  decode step in an nsys trace.
- Change: a Triton split-K bf16 GEMM for M ≤ 16.
- Keep when porting: the Triton path runs only for bf16, 2-D input, no bias and M ≤ 16;
  everything else calls `F.linear`.
- Measured: GPU unit test at M=8: max abs error vs fp32 0.4999 (`F.linear`: 0.5000). Before
  E006c the compiled graph never called it (see E006c).
- Written by: Grok.

### E006b GDN spec decode path uses views instead of copies

- Switch: none; on when applied.
- Files: `qwen_gdn_linear_attn.py` (split packed qkv into `(1, seq, heads, dim)` views),
  `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py` (token and head
  strides address non-packed views).
- Problem: the spec decode path made contiguous copies of q, k, v each step.
- Change: pass strided views; the fused sigmoid gating kernel takes the strides.
- Keep when porting: the KDA path keeps packed pointer math (its callers pass contiguous
  tensors); the last-dim stride must be 1 (unit-stride vector load).
- Measured: together with E006a/c, see the table above.
- Written by: Grok.

### E006c Run-time dispatch for the small-M GEMM

- Switch: `VLLM_E006_SMALL_M=0` routes back to `F.linear`, read inside the op at run time.
- Files: `vllm/model_executor/layers/mamba/gdn/small_m_bf16_gemm.py` (adds the custom op
  `torch.ops.vllm.small_m_bf16_linear`), `linear.py`, `qwen_gdn_linear_attn.py`.
- Problem: E006a's `x.shape[0] <= 16` check sat in `apply`. torch.compile traces the general
  graph at `max_num_batched_tokens` with symbolic shapes, so the check was decided at trace
  time and the compiled graph always called `F.linear` (no `_small_m_splitk_kernel` in the
  nsys trace).
- Change: register the GEMM as a custom op that decides at run time, so each CUDA graph
  captured per batch size takes the Triton path when M ≤ 16. The switch is read inside the
  op because 0.33's AOT compile cache keys on source, not on environment variables; a switch
  read at trace time would be frozen into a cached graph.
- Keep when porting: the op's fake implementation returns shape (M, N); the dispatch and the
  switch stay inside the op.
- Signs it is off or broken: in nsys `in_proj_ba` still shows
  `cutlass_80_wmma_tensorop_bf16_s161616gemm`; decode step back near 21.2 ms.
- Measured (2026-11-04): decode step 21.15–21.4 (switch off) → 19.74–19.91 ms (on); decode
  logit KL vs off 4.1e-4, against a same-build run-to-run floor of 1.7e-4 under
  concurrency. Rebuilding the tree or the compile cache alone gives about 4.5e-4.
- Written by: Claude.

### E031 64-token prefix-cache hits on the hybrid stack (plus a CoW block-copy fix)

- Switch: off unless both `VLLM_E031_SWA_FINE_HITS=1` and `--prefix-match-unit 64` are set.
  `VLLM_E031_DEBUG=1` adds hit / copy-on-write / tail-key / Mamba-slot logs and a byte check of
  every per-layer tensor after the first `VLLM_E031_DEBUG_VERIFY_N` (40) block copies.
- Files: `vllm/v1/core/single_type_kv_cache_manager.py` (`SlidingWindowManager` fine-grained
  lookup and prompt-tail partial key), `vllm/v1/worker/utils.py` (`copy_kv_cache_blocks_inplace`),
  `vllm/v1/worker/mamba_utils.py` (debug log only).
- Problem: agent-loop TTFT cycled between about 200 and 900 ms with the distance of the previous
  prompt from a multiple of the 1600-token block, because prefix hits landed only on the block
  grid. vLLM 0.33 has `--prefix-match-unit` with prompt-tail Mamba states, but fine-grained hits
  were switched off because the DFlash drafter's sliding-window manager does not support them.
  Turning them on exposed a second bug: a hit that resumes mid-block copies the tail block
  first, and the copy de-duplicated per-layer tensors by start address. With the E005 grouping
  the GDN state view and the attention/drafter KV view of one layer slot start at the same
  address, so the KV view was skipped and the resumed request read partly zeroed KV (KL vs a
  full recompute 1.2e-2, 13x the concurrency noise floor).
- Change: no kernel change. The scheduler already splits the prefill at the prompt's last
  64-token boundary and keeps the Mamba state there; the drafter's sliding-window group now
  searches final lengths on the 64-token grid and registers the prompt-tail key like full
  attention. The block copy de-duplicates by (address, shape, stride, dtype).
- Keep when porting: with the switch and flag unset the behaviour is the stock one (the copy
  fix only runs when copy-on-write happens). DFlash does not drop an EAGLE block, so tail keys
  and Mamba tail states sit at floor((P-1)/64)*64. Conversations that rewrite earlier messages
  fall back to the block grid.
- Measured (2x RTX 3090, replay of four stored agent chains, max_tokens 1): 29.5K-token chains
  with a fixed system prompt and tools p50 796 / 805 -> 230 / 229 ms, p90 1,247 / 948 -> 538 /
  497 ms; rewriting and short chains unchanged. After the fix: the same prompt sent twice
  (second send resumes mid-block) is bitwise identical; a mid-block GPU hit extended by LMCache
  chunks is bitwise identical to a fresh recompute; 64-grid hits vs recompute KL 6.7e-4 against
  a 9.4e-4 concurrency noise floor; drafter acceptance 0.242 vs 0.241.
- Written by: Claude (host).

