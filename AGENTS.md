# AGENTS.md

Entry point for coding agents. This repo holds model-specific patches for
[vLLM](https://github.com/vllm-project/vllm) and [LMCache](https://github.com/LMCache/LMCache),
one Qwen model at a time, plus the method we used to make and measure them.

- It is a small, specialized project and an index, not a fork. Each patch set is made for
  one model on one rig and makes no attempt at generality. Other models, GPUs,
  quantizations or parallel sizes are not tested, and supporting them is out of scope.
- Patches are kept per model, per engine, per engine version. A patch is a `git diff`
  against the untouched files of an upstream release tag.

| Model | Folder | Engines | Status |
|---|---|---|---|
| Qwen3.8-27B (48 GDN + 16 attention layers), AWQ-INT4, DFlash2 drafter, 2× RTX 3090, TP2 | [qwen3.8-27b](qwen3.8-27b/README.md) | vLLM 0.33.0, 0.28.0; LMCache 0.5.6 | in our production since 2026-11-04 |

What the Qwen3.8-27B set did on our rig (one request, greedy; details in
[results](qwen3.8-27b/results/README.md)): decode step 27.10–27.47 ms on unpatched vLLM
0.33.0 → 19.74–19.91 ms with the series, KV pool 394,971 → 488,000 tokens, LMCache L2 bytes
−43% with E014. Whole-stack benchmark tables live in
[vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes).

## Routes

Pick one route, load its skill, and follow it.

| You want to | Load | Then read |
|---|---|---|
| Apply a patch set to an installed engine or a source checkout, our model and rig | [skills/apply-patch-set](skills/apply-patch-set/SKILL.md) | the model README, the version folder's cards |
| Port the approach to your own GPUs, model, drafter or quantization | [skills/port-to-your-hardware](skills/port-to-your-hardware/SKILL.md) | its derive-constants reference, the cards' "Keep when porting" lines |
| Research a new model or engine version the way we did | [skills/optimization-method](skills/optimization-method/SKILL.md) | [skills/measurement](skills/measurement/SKILL.md), [skills/engine-internals](skills/engine-internals/SKILL.md) |

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
skills/measurement/scripts/             step_probe.py, logit_diff.py, example prompt files
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
  named and the tree is unchanged; fix it by hand against that patch's card.
- After applying, set the switches from the model README's patch table and confirm each
  patch from the log line its card names. Steps, backup and rollback:
  [skills/apply-patch-set](skills/apply-patch-set/SKILL.md).
- With a switch unset, E003, E005 and E014 run upstream code. E006a/b/c are on when
  applied; `VLLM_E006_SMALL_M=0` turns E006a/c off.

## How patches are checked

- Every series applies with zero fuzz to its upstream tag (`check-upstream.sh`). The
  model README gives the date of the last check (2026-11-07 for Qwen3.8-27B) and whether
  the patched files equal the ones our production engine runs.
- Drafter-only and KV-layout changes: greedy outputs compared with the unpatched engine,
  reported as k/N identical, with acceptance listed separately.
- Changes to the target model's math (kernels, all-reduce, engine versions): decode-logit
  KL over the top-20 tokens against the unpatched path, judged against the same build's
  run-to-run KL under one concurrent request as the noise floor. We do not expect
  bit-identical tokens across builds or kernels.
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
```

Name who wrote a patch only the way the existing cards do.

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
- One direction per sentence, in positive form.
- Leave out local paths, host names, private service ports and account details.

License: Apache-2.0, the license of vLLM and LMCache; the original code keeps its
authors' copyright. Patches were written with AI coding agents and reviewed, tested and
measured by us; each card names who wrote it.
