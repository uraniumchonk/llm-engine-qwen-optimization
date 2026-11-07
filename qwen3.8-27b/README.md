# Qwen3.8-27B on 2× RTX 3090

vLLM and LMCache patches for Qwen3.8-27B, a hybrid Gated DeltaNet (GDN) model, served
with a DFlash2 speculative drafter. In our production since 2026-11-04.

Agents start at [AGENTS.md](../AGENTS.md). Every experiment behind these patches:
[results](results/README.md); where a decode step's time goes:
[step-anatomy](results/step-anatomy.md).

## Scope: one model, one rig

These patches are designed for this one model on this one machine and make no attempt
at generality.

- Model: Qwen3.8-27B (48 GDN layers + 16 full-attention layers), AWQ-INT4, with the
  DFlash2 W4A16 drafter (5 sliding-window layers, 7 speculative tokens).
- Machine: 2× RTX 3090 (sm86), tensor parallel 2.

The design choices follow from that setup:

- E003 exists because the 3090 lacks native fp8 and our drafter runs on Triton.
- E005's group size 16 is chosen for layer counts 48 / 16 / 5.
- E006 is tuned for the `in_proj_ba` shape of this model during spec decode (M ≤ 16,
  N = 48, K = 5120), and E006b changes only the Qwen GDN path.
- E014 assumes GDN groups, one attention group and one drafter sliding window.

We test no other model, drafter, GPU, quantization or parallel size, and we do not
support them. To port, use the [port skill](../skills/port-to-your-hardware/SKILL.md): it
gives the method and the constants to re-derive, and the cards describe each change in
enough detail to port it yourself.

## Index

| Engine | Folder | Patches | Tested |
|---|---|---|---|
| vLLM 0.33.0 | [`vllm/v0.33.0`](vllm/v0.33.0/README.md) | E003, E005, E006a, E006b, E006c | in our production since 2026-11-04 |
| vLLM 0.28.0 | [`vllm/v0.28.0`](vllm/v0.28.0/README.md) | E003 | A/B on 2026-11-03 |
| LMCache 0.5.6 | [`lmcache/v0.5.6`](lmcache/v0.5.6/README.md) | E014, plus kv_probe (test only) | in our production since 2026-11-04 |

| Patch | What it does | Switch | Measured on our rig |
|---|---|---|---|
| E003 | Triton attention reads fp8 e4m3 KV on pre-SM89 GPUs (software decode), so the drafter can leave FlashInfer and its per-step CPU sync | `VLLM_TRITON_FP8_SW_DEQUANT=1` | decode step 27.10–27.47 → 21.77–22.10 ms (0.33, greedy) |
| E005 | Forces the hybrid KV cache group size instead of taking the smallest layer count (5 drafter layers) | `VLLM_KV_GROUP_SIZE=16` | KV pool 396,364 → 488,000 tokens (+23.1%) |
| E006a/b/c | Small-M split-K Triton GEMM for GDN `in_proj_ba` during spec decode, dispatched at run time; GDN spec path uses views instead of copies | on when applied; `VLLM_E006_SMALL_M=0` turns E006a/c off | decode step 21.15–21.40 → 19.74–19.91 ms |
| E014 | LMCache stores GDN and drafter state only at checkpoint chunks; attention KV is stored for every chunk | `LMCACHE_MEOW_CKPT_K=4` | L2 bytes −43% on our workload |

With the switch unset, E003, E005 and E014 leave upstream behaviour unchanged.

## Test rig

| | |
|---|---|
| GPUs | 2× RTX 3090 (sm86), tensor parallel 2 over PCIe, 250 W limit |
| Target | Qwen3.8-27B-AWQ-INT4: 48 GDN layers + 16 attention layers, `--attention-backend FLASHINFER` |
| Drafter | DFlash2-W4A16-GPTQ (5 sliding-window layers), `method=dflash`, 7 speculative tokens, `attention_backend=TRITON_ATTN` |
| vLLM flags | `--kv-cache-dtype fp8 --block-size 1600 --mamba-ssm-cache-dtype bfloat16 --max-num-seqs 4 --max-num-batched-tokens 2048 --enable-prefix-caching` |
| LMCache | multiprocess server via `LMCacheMPConnector` (`kv_both`), `--chunk-size 1600 --engine-type default --separate-object-groups` |
| Stack (0.33) | torch 2.14.0+cu132, Triton 3.8.0, FlashInfer 0.7.2.post1 |

Measurements are single runs or short repeated runs on this one rig, under the dates in
each card. The cards give ranges, not means.

## Checked on 2026-11-07

- Every series applies with zero fuzz to its upstream tag (`scripts/check-upstream.sh`:
  vLLM v0.33.0, v0.28.0, LMCache v0.5.6).
- Applying the series to the released package files gives the files our production
  engine runs, byte for byte (vLLM 0.33.0 PyPI wheel, 7 files; LMCache 0.5.6 built from
  the PyPI sdist, 2 files).

Field notes on the same model and rig (version matrix, failure modes):
[vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes).
