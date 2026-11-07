# vLLM 0.28.0

Upstream: `vllm-project/vllm` tag `v0.28.0`. Apply order: [`series`](series).
Tested on 2026-11-03; we moved to [vLLM 0.33.0](../v0.33.0/README.md) the next day, where
this patch has a port and E005/E006 join it. Rig: [model README](../../README.md#test-rig).
Scope: written for Qwen3.8-27B (48 GDN + 16 attention layers) with the DFlash2 drafter on
2× RTX 3090 only; no attempt at generality. See [Scope](../../README.md#scope-one-model-one-rig).

Our 0.28.0 production also carried backports of upstream PRs
[#54165](https://github.com/vllm-project/vllm/pull/54165) and
[#50885](https://github.com/vllm-project/vllm/pull/50885); they are not in this folder.
E003 applies to plain v0.28.0 as well as on top of them (checked 2026-11-07).

### E003 Triton fp8 e4m3 KV on pre-SM89 GPUs

- Switch: `VLLM_TRITON_FP8_SW_DEQUANT=1`. Off by default; when off, or on SM89 and newer,
  the code path is upstream's.
- Files: `vllm/v1/attention/backends/triton_attn.py`,
  `vllm/v1/attention/ops/triton_unified_attention.py`.
- Problem, change, what to keep and failure signs: same as the
  [0.33.0 card](../v0.33.0/README.md#e003-triton-fp8-e4m3-kv-on-pre-sm89-gpus).
- Measured (2026-11-03, GPU A/B): decode step 27.6–31.0 → 22.5–23.0 ms, KV pool 369,545
  tokens in both, greedy outputs 6/6 identical. The baseline ran the drafter on FlashInfer.
- Written by: Claude, with a hardening pass by Codex (NaN bytes 0x7F/0xFF decode to NaN
  instead of ±480; `tl.static_assert` rejects an fp8 query).
