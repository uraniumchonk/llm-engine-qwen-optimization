# AGENTS.md

Entry point for coding agents. This repo holds model-specific patches for
[vLLM](https://github.com/vllm-project/vllm) and [LMCache](https://github.com/LMCache/LMCache),
one Qwen model at a time, plus the method we used to make and measure them.

- It is a small, specialized project and an index, not a fork. Each patch set is made for
  one model on one rig and makes no attempt at generality. Other models, GPUs,
  quantizations or parallel sizes are untested and we do not support them. Porting to
  them is your own work: the port route below gives the method and the constants to
  re-derive.
- Patches are kept per model, per engine, per engine version. A patch is a `git diff`
  against the untouched files of an upstream release tag.

| Model | Folder | Engine version | Patches | Status |
|---|---|---|---|---|
| Qwen3.8-27B (48 GDN + 16 attention layers), AWQ-INT4, DFlash2 drafter, 2× RTX 3090, TP2 | [qwen3.8-27b](qwen3.8-27b/README.md) | vLLM 0.33.0 | E003, E005, E006a/b/c, E031 | in our production since 2026-11-04 (E031 since 2026-11-13) |
| | | LMCache 0.5.6 | E014, E030 (+ kv_probe, test only) | in our production since 2026-11-04, with vLLM 0.33.0 (E030 since 2026-11-13) |
| | | vLLM 0.28.0 | E003 | A/B tested 2026-11-03; superseded by 0.33.0 |

What the Qwen3.8-27B set did on our rig (one request, greedy; details in
[results](qwen3.8-27b/results/README.md)): decode step 27.10–27.47 ms on unpatched vLLM
0.33.0 → 19.74–19.91 ms with the series, KV pool 394,971 → 488,000 tokens (both exact
start-log readings), LMCache L2 bytes −43% with E014; prefix hits on a 64-token grid
with E031 (agent turns with a fixed system prompt and tools, 29.5K tokens: TTFT p50 796 →
230 ms). Whole-stack benchmark tables live in
[vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes).

## Guardrails

Follow these on every route.

1. Brief your human first. Before you change anything, tell the person you work for in a
   few plain lines: what this repo is and the one setup it was made for, how their
   setup compares, which route you plan and why, what you will touch, what to back up
   first (guardrail 2), and what you need from them (GPU time, permission to stop a
   server). Ask which route they want when their request does not say. Start after they
   agree. Example:

   > This repo has vLLM and LMCache patches made for one setup: Qwen3.8-27B with a
   > DFlash2 drafter on 2× RTX 3090. There they cut the decode step from 27.1–27.5 ms to
   > 19.7–19.9 ms. You run Qwen3.8-27B on one RTX 4090, so the patches will not apply as
   > is. I plan the port route: inventory your setup, decide per patch, and measure your
   > baseline. I will work in a separate venv and leave your running server alone. I need
   > about an hour of GPU time with the server stopped. Go ahead?

2. Work on a copy. Apply patches to a separate venv or a copy of `site-packages`, never to
   the `site-packages` a running production server imports. Move a patch set to
   production only after its numeric check passes, with a backup and a rollback path.
   Before that move, tell your human what a full snapshot covers and let them decide
   whether to take one: the whole venv (a self-built LMCache included), the start command
   and environment variables, and the driver and kernel state. `pre-patch-backup.tgz`
   from apply-patch-set holds only the files the series touches.
3. Stop and report on CONFLICT. Leave the `.patch` files and the target files as they
   are. A raised fuzz factor or a hand edit that forces a patch in is not a fix. Porting
   to another engine version is a new version folder, made with the optimization-method
   skill.
4. Report "not checked" when `check-upstream.sh` cannot reach GitHub. Only a run whose
   every line reads OK counts as checked.
5. Keep patch ids and measured numbers as written. A correction is a new dated line next
   to the old number; an id is never reused.
6. Ask the operator before stopping, restarting or reconfiguring a production server.
7. End with a report:
   - the version folder, its upstream tag, and the installed versions;
   - the `apply.sh --check` output, and the `check-upstream.sh` result or "not checked";
   - per patch, the start-log line found, or "missing";
   - the numeric check per class: step-time range with n, greedy outputs identical k/N,
     KL with its noise floor and token counts, `kv_probe` digests equal k/N;
   - every environment variable and flag you set, and how to undo each.

## Routes

Pick one route, load its skill, and follow it.

| You want to | Load | Then read |
|---|---|---|
| Apply a patch set to an installed engine or a source checkout, our model and rig | [skills/apply-patch-set](skills/apply-patch-set/SKILL.md) | the model README, the version folder's cards |
| Port the approach to your own GPUs, model, drafter or quantization | [skills/port-to-your-hardware](skills/port-to-your-hardware/SKILL.md) | its derive-constants reference, the cards' "Keep when porting" lines |
| Research a new model or engine version the way we did | [skills/optimization-method](skills/optimization-method/SKILL.md) | [skills/measurement](skills/measurement/SKILL.md), [skills/engine-internals](skills/engine-internals/SKILL.md) |
| Answer a question about this stack | [Answering questions](#answering-questions) below | the file its table names |

Read order for any route:

1. This file.
2. The model README: [qwen3.8-27b/README.md](qwen3.8-27b/README.md) (scope, rig, flags,
   patch summary, last check).
3. The version folder's cards: [vllm/v0.33.0](qwen3.8-27b/vllm/v0.33.0/README.md),
   [vllm/v0.28.0](qwen3.8-27b/vllm/v0.28.0/README.md),
   [lmcache/v0.5.6](qwen3.8-27b/lmcache/v0.5.6/README.md).
4. The route's skill, then the references it names.
5. Before you compare any number with ours: [skills/measurement](skills/measurement/SKILL.md)
   and [qwen3.8-27b/results](qwen3.8-27b/results/README.md).

## Answering questions

This stack was studied down to each kernel and the GPU feature it runs on: tensor-core
modes, fp8 support, tile shapes, PCIe peer access, what torch.compile freezes at trace
time. At that depth, general knowledge about vLLM or GPUs is often wrong for this setup,
and a wrong answer sounds as sure as a right one. Answer from the repo:

1. Find the fact in the file the table below names, and read the run it cites.
2. Quote what was measured with its experiment id, setup, engine version and date.
   Label an estimate "estimate, not measured" and give its arithmetic.
3. When the repo has no measurement for the question, say "not measured here" and name
   the measurement that would answer it ([measurement](skills/measurement/SKILL.md)). A
   different GPU, model, quantization or engine version counts as not measured.
4. Check claims about code against the source of the installed version.

| Question | Read |
|---|---|
| Where does a decode step's time go? | [results/step-anatomy.md](qwen3.8-27b/results/step-anatomy.md) |
| Was X tried, and what happened? | [results/README.md](qwen3.8-27b/results/README.md), dead ends included |
| Why does a patch exist, and what must stay when porting it? | the patch's card in its version folder |
| Does a patch transfer to my GPU or model? | [port skill](skills/port-to-your-hardware/SKILL.md) §2, [05-ampere-sm86](skills/engine-internals/references/05-ampere-sm86.md) |
| What does this layer, flag or backend do here? | [engine-internals](skills/engine-internals/SKILL.md) index, then its reference |
| Is my number comparable to yours? | [measurement](skills/measurement/SKILL.md) rules, [traps](skills/measurement/references/traps.md) |

**Assume you do not know.** Treat your own knowledge of kernels and GPU features as
unverified for this stack. When the repo does not settle a question, tell your human you
have not verified it. Quantization is the most common trap: a scheme's name says how
weights and activations are stored, not what the tensor cores compute.

- W4A16 and W8A16 (int4 or int8 weights, 16-bit activations): the kernel unpacks the
  weights in registers and multiplies in bf16/fp16, so int8 tensor cores stay idle; the
  gain is memory bandwidth. vLLM 0.33's Marlin takes 4-bit and 8-bit weights from SM75 up
  (`query_marlin_supported_quant_types` in `marlin_utils.py`); our W4A16 runs on Marlin
  (E022).
- W8A8 int8: activations are quantized to int8 as well, and vLLM 0.33 routes the layer to
  cutlass int8 `scaled_mm`, which uses int8 tensor cores on Ampere (code read, not
  measured here).
- W8A8 fp8 below SM89, such as a 3090: vLLM 0.33 logs "Your GPU does not have native
  support for FP8 computation ... Weight-only FP8 compression will be used leveraging the
  Marlin kernel" (`marlin_utils_fp8.py`) and runs fp8 weights with 16-bit math.
- int8 vs fp8 activations: which loses more accuracy depends on the model and its
  calibration; we have not measured it on this model.
- fp8 KV cache on sm86 is a storage format; the math stays 16-bit (E003).
- Evidence of the path that ran is the start-log kernel line or warning and the kernel
  names in an nsys trace. A scheme name, a config field or a model card shows the storage
  format only.

Answers that memory gets wrong on this stack, all measured:

- "Triton can use fp8 KV on a 3090." Triton 3.7.1 could not compile fp8e4nv loads or
  stores on sm86; E003 decodes the bytes in software.
- "A faster kernel makes the step faster." E006a's kernel passed its unit test, yet
  torch.compile fixed the branch at trace time and the compiled graph never called it
  (E008); E006c moved the branch into a custom op.
- "nsys shows where the time goes." Node-level tracing stretched a step from 27.6 to
  35.7 ms (vLLM 0.28.0) and the graph launch call from 58 to 954 µs (vLLM 0.33.0).
- "Custom all-reduce runs on four GPUs." vLLM disables it on more than two PCIe-only GPUs.
- "Speculative decoding always helps." Four prose streams on vLLM 0.33.0 ran 168.5 tok/s
  with DFlash2 and 240.8 tok/s without (E006).

## Repo map

```
AGENTS.md                               this file
README.md                               short landing page for people
LICENSE                                 Apache-2.0
scripts/apply.sh                        apply one version folder to a tree (zero fuzz, scratch copy first)
scripts/check-upstream.sh               apply one version folder to the upstream tag's files (needs curl)
<model>/README.md                       scope, test rig and flags, patch index, date of the last check
<model>/<engine>/v<X.Y.Z>/UPSTREAM      repo= and tag= of the upstream release
<model>/<engine>/v<X.Y.Z>/series        patch file names in apply order (# starts a comment)
<model>/<engine>/v<X.Y.Z>/*.patch       git diffs against the untouched upstream files of that tag
<model>/<engine>/v<X.Y.Z>/README.md     one card per patch
<model>/<engine>/v<X.Y.Z>/test-only/    patches used only for testing, not in series
<model>/results/README.md               every experiment: question, outcome, deciding numbers
<model>/results/step-anatomy.md         where one decode step's time goes
skills/<name>/SKILL.md                  agent skills (frontmatter name + description: when to load)
skills/<name>/references/               details a skill points to
skills/measurement/scripts/             step_probe.py, logit_diff.py, typev2_bench.py, example prompt files
```

Patch ids (E003, E014, ...) are our experiment numbers, unique across the repo. A patch
keeps its id when ported to a newer engine version; letters split one experiment into
several patches (E006a/b/c). The results page uses the same ids.

## Apply and check

Paths inside the patches are relative to the package root (`a/vllm/...`,
`a/lmcache/...`). That root is the top of a source checkout at the tag in `UPSTREAM`, and
also the `site-packages` directory of the installed PyPI wheel.

```bash
scripts/apply.sh qwen3.8-27b/vllm/v0.33.0 /path/to/site-packages --check   # CLEAN / APPLIED / CONFLICT
scripts/apply.sh qwen3.8-27b/vllm/v0.33.0 /path/to/site-packages           # apply the whole series
scripts/check-upstream.sh qwen3.8-27b/vllm/v0.33.0                          # every line must read OK
```

- `apply.sh` needs GNU patch. It tries the whole series on a scratch copy of the touched
  files first, because later patches build on earlier ones (E006c edits lines E006a
  adds). CLEAN: the series applies in order, and apply mode then patches the tree.
  APPLIED: the tree already holds the whole series. CONFLICT: the first failing patch is
  named and the tree is unchanged; stop and report (guardrail 3).
- After applying, set the switches from the model README's patch table and confirm each
  patch from the log line its card names. Steps, backup and rollback:
  [skills/apply-patch-set](skills/apply-patch-set/SKILL.md).
- With a switch unset, E003, E005 and E014 run upstream code, except that E014 still runs
  role detection at connector start ([card](qwen3.8-27b/lmcache/v0.5.6/README.md)).
  E006a/b/c are on when applied; `VLLM_E006_SMALL_M=0` turns E006a/c off.

## How patches are checked

- Every series applies with zero fuzz to its upstream tag (`check-upstream.sh`). The
  model README gives the date of the last check (2026-11-07 for Qwen3.8-27B) and whether
  the patched files equal the ones our production engine runs.
- Drafter-only and KV-layout changes: greedy outputs compared with the unpatched engine,
  reported as k/N identical, with acceptance listed separately.
- Changes to the target model's math (kernels, all-reduce): decode-logit KL over the
  top-20 tokens against the unpatched path, written next to two reference levels measured
  on vLLM 0.33.0 (2026-11-04): 1.72e-4 (same build, one extra concurrent request, 1,363
  tokens) and 4.48e-4 (same code path, other tree, fresh compile cache; one pair, 505
  tokens). The patches we adopted measured 4.11e-4 to 4.64e-4: 2.4–2.7× the first level
  and about equal to the second. We do not expect bit-identical tokens across builds or
  kernels.
- Engine upgrade from vLLM 0.28.0 to 0.33.0: no KL was recorded; greedy text was 2/6
  identical (E004). The 0.28.0 stack has typev2 speed runs and a corruption soak (360/360
  agent turns clean with LMCache retrieve on, 2026-08-28) in the notes repo. We found no
  output-quality comparison between the two versions (checked 2026-11-09).
- KV cache store changes: GPU checksums of every chunk on store and retrieve, zero
  mismatches (for LMCache: `qwen3.8-27b/lmcache/v0.5.6/test-only/kv_probe.patch`).
- A speed number without its correctness check does not make a patch adopted.

Tools and the eight measurement rules: [skills/measurement](skills/measurement/SKILL.md).

## Adding a model or an engine version

1. A new model gets `<model>/README.md` with its scope and test rig (copy the layout of
   [qwen3.8-27b/README.md](qwen3.8-27b/README.md)) and a row in the model table of this
   file.
2. A new engine version gets `<model>/<engine>/v<X.Y.Z>/` with `UPSTREAM`, `series` and
   the patch files. Make each patch as a `git diff` against the untouched upstream files
   of that tag, one patch per change.
3. Name a patch `<id>-<short-name>.patch`. The id is the experiment number, unique across
   the repo, and stays the same when the patch is ported.
4. Write a card per patch in the version folder's `README.md` (template below) and add
   the version to the model README's index.
5. Run `scripts/check-upstream.sh <version folder>`; every line must read OK.
6. Add the experiment's row to `<model>/results/README.md`: question, outcome (adopted,
   dead, blocked, open, measured), deciding numbers with units, date.

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
- Written by:
```

`Written by` names who wrote the patch and who hardened or ported it, for example
"Codex; hardened by Claude".

## Writing rules for this repo

Every file here is read by other agents; write facts only.

- Keep three kinds of sentence apart. Measured: what, setup, n, date, range. Estimate:
  label it "estimate, not measured" and give the arithmetic and its inputs. Decision:
  what was chosen and the measured reason.
- Ranges over means; units on every number; the engine version a number was measured on.
- Frequencies as k/N; a partial check says how much was checked.
- "Not adopted" is not "no effect": write the n and the setup it was measured on.
- A design statement ("removes a sync") and a measured effect ("1.3 ms shorter") are two
  sentences; the second needs its own run.
- Documents describe what the code does as checked against the code or a run.
- Keep superseded numbers and add a dated correction next to them.
- One direction per sentence, in positive form. Write "On CONFLICT, stop and report."
  rather than "Don't fix a CONFLICT yourself, unless the change looks small."
- Leave out local paths, host names, private service ports and account details.

## Requirements

- bash 4 or newer and GNU patch 2.7 or newer for `apply.sh` (tested with bash 5.2 and
  patch 2.7.6).
- curl and network access to raw.githubusercontent.com for `check-upstream.sh`.
- Python 3 for the measurement scripts (tested with 3.12.3); `step_probe.py` needs
  `httpx`, `logit_diff.py` and `typev2_bench.py` use the standard library only.
- A vLLM or LMCache install at the exact version of the folder you apply.

License: Apache-2.0, the license of vLLM and LMCache; the original code keeps its
authors' copyright. The notes repo (vllm-hybrid-mamba-notes) is MIT and holds numbers and
links; patch code stays here under Apache-2.0. Patches were written with AI coding agents and reviewed, tested and
measured by us; each card names who wrote it.
