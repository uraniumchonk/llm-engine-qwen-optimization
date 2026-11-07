# llm-engine-qwen27b-optimization

Patches we run on top of [vLLM](https://github.com/vllm-project/vllm) and
[LMCache](https://github.com/LMCache/LMCache) to serve Qwen3.8-27B, a hybrid Gated
DeltaNet (GDN) model, with a DFlash2 speculative drafter on two RTX 3090s.

## Scope: one model, one rig

These patches are designed for one model on one machine and make no attempt at
generality.

- Model: Qwen3.8-27B (48 GDN layers + 16 full-attention layers), AWQ-INT4, with the
  DFlash2 W4A16 drafter (5 sliding-window layers, 7 speculative tokens).
- Machine: 2× RTX 3090 (sm86), tensor parallel 2.

The design choices follow from that setup:

- E003 exists because the 3090 lacks native fp8 and our drafter runs on Triton.
- E005's group size 16 is chosen for layer counts 48 / 16 / 5.
- E006 is tuned for the `in_proj_ba` shape of this model during spec decode (M ≤ 16,
  N = 48, K = 5120), and E006b changes only the Qwen GDN path.
- E014 assumes GDN groups, one attention group and one drafter sliding window.

We test no other model, drafter, GPU, quantization or parallel size. Other setups are
out of scope, including requests to support them. The cards describe each change in
enough detail to port it yourself.

This repo is an index, not a fork. Each engine version has one folder holding the patch
files, the order they apply in (`series`), the upstream tag they target (`UPSTREAM`) and
a `README.md` with one card per patch: the problem, the change, what to keep when porting
it, how to tell it is active, and what we measured.

## Index

| Engine | Folder | Patches | Tested |
|---|---|---|---|
| vLLM 0.33.0 | [`vllm/v0.33.0`](vllm/v0.33.0/README.md) | E003, E005, E006a, E006b, E006c | in our production since 2026-11-04 |
| vLLM 0.28.0 | [`vllm/v0.28.0`](vllm/v0.28.0/README.md) | E003 | A/B on 2026-11-03 |
| LMCache 0.5.6 | [`lmcache/v0.5.6`](lmcache/v0.5.6/README.md) | E014, plus kv_probe (test only) | in our production since 2026-11-04 |

| Patch | What it does | Switch | Measured on our rig |
|---|---|---|---|
| E003 | Triton attention reads fp8 e4m3 KV on pre-SM89 GPUs (software decode), so the drafter can leave FlashInfer and its per-step CPU sync | `VLLM_TRITON_FP8_SW_DEQUANT=1` | decode step 27.1–27.9 → 21.8–22.3 ms (0.33) |
| E005 | Forces the hybrid KV cache group size instead of taking the smallest layer count (5 drafter layers) | `VLLM_KV_GROUP_SIZE=16` | KV pool 396,364 → 488,000 tokens (+23.1%) |
| E006a/b/c | Small-M split-K Triton GEMM for GDN `in_proj_ba` during spec decode, dispatched at run time; GDN spec path uses views instead of copies | on when applied; `VLLM_E006_SMALL_M=0` turns E006a/c off | decode step 21.15–21.4 → 19.74–19.91 ms |
| E014 | LMCache stores GDN and drafter state only at checkpoint chunks; attention KV is stored for every chunk | `LMCACHE_MEOW_CKPT_K=4` | L2 bytes −43% on our workload |

With the switch unset, E003, E005 and E014 leave upstream behaviour unchanged.

## Applying

The patches are `git diff` files with paths relative to the package root (`a/vllm/...`,
`a/lmcache/...`). That root is the top of a source checkout at the tag in `UPSTREAM`, and
also the `site-packages` directory of the installed PyPI wheel.

```bash
scripts/apply.sh vllm/v0.33.0 /path/to/site-packages --check   # CLEAN / APPLIED / CONFLICT
scripts/apply.sh vllm/v0.33.0 /path/to/site-packages           # apply the whole series
scripts/check-upstream.sh vllm/v0.33.0                          # fetch the upstream files at the tag and test
```

`apply.sh` first tries the whole series on a scratch copy and touches the tree only when
every patch applies with zero fuzz. It needs GNU patch. `check-upstream.sh` needs curl.

Then set the switches from the table in the engine's environment. The cards list the log
line that confirms each patch is active.

## How these were checked

Checked on 2026-11-07:

- Every series applies with zero fuzz to its upstream tag (`check-upstream.sh`: vLLM
  v0.33.0, v0.28.0, LMCache v0.5.6).
- Applying the series to the released package files gives the files our production
  engine runs, byte for byte (vLLM 0.33.0 PyPI wheel, 7 files; LMCache 0.5.6 built from
  the PyPI sdist, 2 files).

When a patch was made:

- Numeric kernel changes were compared through the decode logit distribution against the
  unpatched path, using the run-to-run KL of the unpatched build as the noise floor. We do
  not expect bit-identical tokens across builds or kernels. Greedy outputs are reported
  where we ran them.
- LMCache changes were checked with GPU checksums of every chunk on store and retrieve
  (`lmcache/v0.5.6/test-only/kv_probe.patch`).

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

## Adding a version

1. Make `<engine>/v<X.Y.Z>/` with `UPSTREAM` (`repo=` and `tag=`), the patch files and
   `series` (one file name per line, in apply order; `#` starts a comment).
2. Make each patch as a `git diff` against the untouched upstream files of that tag. One
   patch per change. A patch that only serves testing goes in `test-only/` and stays out of
   `series`.
3. Name a patch `<id>-<short-name>.patch`. The id (E003, E014, …) is our experiment number
   and stays the same when the patch is ported to a newer version.
4. Write a card per patch in the folder's `README.md` (template below).
5. Run `scripts/check-upstream.sh <folder>`; every line must read OK.
6. Add the folder to the index above.

Card template:

```markdown
### E### title
- Switch:
- Files:
- Problem:
- Change:
- Keep when porting:
- Signs it is off or broken:
- Measured (date, setup):
- Upstream status:
```

## Provenance and license

The patches were written with AI coding agents (OpenAI Codex, xAI Grok, Anthropic
Claude), then reviewed, tested and measured by us on the rig above. The cards name who
wrote each one.

Apache License 2.0, the license of vLLM and LMCache. The patches modify files of those
projects; the original code keeps its authors' copyright.

Field notes on the same stack (version matrix, failure modes):
[vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes).
