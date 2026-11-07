---
name: apply-patch-set
description: Load when you apply a model folder of this repo (for example qwen3.8-27b/vllm/v0.33.0 or qwen3.8-27b/lmcache/v0.5.6) to an installed engine or a source checkout. Covers matching the version, backing up, apply.sh check and apply, setting the switches, confirming each patch from its log line, the numeric check against the unpatched engine, and rollback.
---

# Apply a patch set

The patch sets are made for Qwen3.8-27B + DFlash2 on 2× RTX 3090
([scope](../../qwen3.8-27b/README.md)). On another model or GPU, read
[port-to-your-hardware](../port-to-your-hardware/SKILL.md) first. Commands below run from
the repo root.

## 1. Match the folder to your install

- Each version folder holds `UPSTREAM` (`repo=`, `tag=`), `series` (apply order) and the
  `.patch` files. Use the folder whose tag equals your installed version exactly:
  `python3 -m pip show vllm lmcache` (or `uv pip show`).
- The tree root is the directory that contains `vllm/` or `lmcache/`: the
  `site-packages` of the installed wheel, or a source checkout at the tag.
- vLLM and LMCache are separate folders; apply each to its own package. Our production
  runs [vllm/v0.33.0](../../qwen3.8-27b/vllm/v0.33.0/README.md) together with
  [lmcache/v0.5.6](../../qwen3.8-27b/lmcache/v0.5.6/README.md).
- LMCache 0.5.6 under torch 2.14: the PyPI wheel's CUDA extension fails to import and
  LMCache falls back to plain torch copies. Build 0.5.6 from the PyPI sdist against your
  torch first, then apply the patches to that install (error text in the LMCache card).
- `scripts/check-upstream.sh <version folder>` (needs curl and network access to
  raw.githubusercontent.com) fetches the tag's files and checks the series against them;
  every line must read OK. `NOT CHECKED` (exit 3) or no network: report the series as
  not checked.

## 2. Back up, check, apply

Apply to a separate venv or a copy of `site-packages` first, never to the one a running
production server imports. Production gets the patch set after step 5 passes, with the
operator's go-ahead.

```bash
V=qwen3.8-27b/vllm/v0.33.0
ROOT=/path/to/test-venv/site-packages   # the directory that contains vllm/

# back up every existing file the series touches (new files have no backup)
(cd "$V" && grep -h '^+++ b/' $(grep -v -E '^(#|$)' series) | sed 's#^+++ b/##' | sort -u) \
  | while read -r f; do [ -e "$ROOT/$f" ] && echo "$f"; done > touched.txt
tar -C "$ROOT" -czf pre-patch-backup.tgz -T touched.txt

scripts/apply.sh "$V" "$ROOT" --check   # CLEAN / APPLIED / CONFLICT, changes nothing
scripts/apply.sh "$V" "$ROOT"           # applies the whole series with zero fuzz
```

- `apply.sh` (needs GNU patch) first runs the whole series on a scratch copy and
  touches the tree only when every patch applies with zero fuzz.
- CLEAN: apply. APPLIED: the tree already holds the series. CONFLICT: the first failing
  patch is named and the tree is unchanged. Stop and report the output. Leave the
  `.patch` files and the target files as they are; a raised fuzz factor or a hand edit
  that forces a patch in is not a fix. A different engine version needs its own version
  folder ([optimization-method](../optimization-method/SKILL.md)).
- The patched source triggers a recompile: vLLM 0.33's compile cache keys on source. Give
  the patched engine its own `VLLM_CACHE_ROOT` if you keep an unpatched one for A/B.
- E005 changes the KV layout. Point LMCache at a fresh L2 directory before the first
  start with it.

## 3. Switches

Set these in the engine's environment and flags; the full flag set of our rig is in
the [model README](../../qwen3.8-27b/README.md#test-rig).

| Patch | Where | Value we run | Unset |
|---|---|---|---|
| E003 | vLLM env | `VLLM_TRITON_FP8_SW_DEQUANT=1`, plus `attention_backend=TRITON_ATTN` in the speculative config and `--kv-cache-dtype fp8` | upstream path |
| E005 | vLLM env | `VLLM_KV_GROUP_SIZE=16` | upstream grouping |
| E006a, E006c | none needed | on when applied; `VLLM_E006_SMALL_M=0` routes back to `F.linear` | — |
| E006b | none | on when applied | — |
| E014 | vLLM env (not the LMCache server) | `LMCACHE_MEOW_CKPT_K=4`; LMCache server with `--separate-object-groups` | every chunk stored |
| kv_probe (test only) | LMCache server env | `LMCACHE_MEOW_KV_PROBE=1` | — |
| upstream switch, not a patch | vLLM env | `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | custom all-reduce |

- The target keeps `--attention-backend FLASHINFER`; only the drafter moves to Triton.
- E014 without `--separate-object-groups` on the server reads the null block id as
  block 0, which is wrong data. Start the server with the flag before setting K.

## 4. Confirm each patch

Start the engine and read its start log. A missing line means the patch is not active,
whatever the step time says.

| Patch | Line to find | If it is missing or wrong |
|---|---|---|
| E003 | `Triton attention: fp8 e4m3 KV via software dequant (E003)`; drafter `Capturing dflash2 CUDA graphs (FULL)` | `FP8 KV cache is not supported by the Triton attention backend` = switch unset; `type fp8e4nv not supported in this architecture` = an fp8 tensor (often the query) reached Triton |
| E005 | `E005: VLLM_KV_GROUP_SIZE overrides 5 -> 16 for layer counts [48, 16, 5]`; `GPU KV cache size` about 488,000 tokens on our rig without LMCache | `VLLM_KV_GROUP_SIZE must be a positive integer`; a divisibility ValueError; no line = another grouping path returned first |
| E006a/c | no log line: in an nsys trace `in_proj_ba` runs `_small_m_splitk_kernel` | `cutlass_80_wmma_tensorop_bf16_s161616gemm` still present; step back near 21.2 ms on our rig |
| E006b | no log line | — |
| E014 | `LMCACHE_MEOW_CKPT_K=4 roles=['gdn', 'gdn', 'gdn', 'attention', 'drafter']` | roles all `attention`: detection failed, every chunk stored; one warning line: an exception, every chunk stored |
| LMCache build | server start log lists kernel groups GDN `hs=560` ×3, attention `kv=2 hs=512`, drafter `nh=4 hs=256` | an `undefined symbol` import error of `lmcache.cuda_ops`: rebuild from the sdist |
| PCIe IPC | `Using ['FLASHINFER_PCIE_IPC', 'CUSTOM', 'PYNCCL'] all-reduce backends` | only `CUSTOM, PYNCCL`: the variable is not in the engine's environment |

Per-patch failure signs in full: the "Signs it is off or broken" line of each card.

## 5. Numeric check

Run on a test instance, the same day, against the unpatched engine with the same flags.
Tools and rules: [measurement](../measurement/SKILL.md).

1. Step time: `step_probe.py --types code,prose,mixed --max-tokens 300 --conc 1 --reps 2`.
   Ours, one request, greedy, without LMCache: 27.10–27.47 ms unpatched 0.33.0, 19.74–19.91
   ms with the full series ([results](../../qwen3.8-27b/results/README.md#adopted-path)).
2. Greedy identity for E003 and E005 (drafter backend and KV layout): the texts that
   `step_probe.py --out` saves should match the unpatched run; ours were 6/6 for each.
3. Decode logits for E006 (target math): `logit_diff.py gen` on both builds and on the
   unpatched build with one extra concurrent request, then `gcmp`. Ours: KL 4.11e-4 for
   E006c off vs on, floor 1.72e-4, about 4.5e-4 between two builds of the same code.
   Judge your KL against your own floor.
4. LMCache with E014: run a workload twice with `kv_probe` on a test server; every
   retrieve digest equals its store digest (ours: 932 of 932). L2 should hold fewer GDN
   than attention objects (ours 44 vs 146 at K=4).

A patch that shows its log line but fails its numeric check stays off.

## 6. Rollback

- Switch level, no reinstall: unset `VLLM_TRITON_FP8_SW_DEQUANT` (and move the drafter
  back to its old backend), unset `VLLM_KV_GROUP_SIZE`, unset `LMCACHE_MEOW_CKPT_K` or set
  0, set `VLLM_E006_SMALL_M=0`. E003, E005 and E014 then run upstream code paths; E006a/c
  route to `F.linear`. E006b has no switch.
- File level: restore the backup and delete the files the series created
  (`small_m_bf16_gemm.py` for vLLM 0.33.0, `meow_checkpoint.py` for LMCache 0.5.6):

```bash
tar -C "$ROOT" -xzf pre-patch-backup.tgz
rm -f "$ROOT/vllm/model_executor/layers/mamba/gdn/small_m_bf16_gemm.py"
scripts/apply.sh "$V" "$ROOT" --check   # must now report CLEAN
```

- Or reinstall the package at the same version.
- KV layout: after E005 is off, the KV layout is upstream's again; L2 objects written with
  group size 16 belong to the other layout. Use the L2 directory that matches the layout
  you run.
- Compile cache: unpatched source compiles again (or reuses the unpatched cache root).

## 7. Report

End with:

- the version folder, its upstream tag, and the installed engine versions;
- the `apply.sh --check` output, and the `check-upstream.sh` result or "not checked";
- per patch, the start-log line found (section 4), or "missing";
- the numeric check per class (section 5): step-time range with n, greedy outputs
  identical k/N, KL with its noise floor and token counts, `kv_probe` digests equal k/N;
- every environment variable and flag you set, and how to undo each (section 6).
