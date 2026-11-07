# llm-engine-qwen-optimization

Model-specific patches for [vLLM](https://github.com/vllm-project/vllm) and
[LMCache](https://github.com/LMCache/LMCache), one Qwen model at a time.

This is a small, specialized project. Each patch set is designed for one model on our
hardware and makes no attempt at generality. This repo is an index, not a fork: patches
are kept per model, per engine, per engine version.

## Models

| Model | Folder | Engines | Status |
|---|---|---|---|
| Qwen3.8-27B (hybrid GDN) + DFlash2 drafter, 2× RTX 3090 | [`qwen3.8-27b`](qwen3.8-27b/README.md) | vLLM 0.33.0, 0.28.0; LMCache 0.5.6 | in our production since 2026-11-04 |

Further Qwen models get their own folder when we start optimizing for them.

## Scope

Each model folder's README names the exact model, drafter, GPUs and engine flags its
patches were made for. A patch is designed for that setup only. We test no other model,
GPU, quantization or parallel size, and requests to support other setups are out of
scope. The cards describe each change in enough detail to port it yourself.

## Layout

```
<model>/README.md                      scope, test rig, patch summary for that model
<model>/<engine>/v<X.Y.Z>/UPSTREAM     repo= and tag= of the upstream release
<model>/<engine>/v<X.Y.Z>/series       patch file names in apply order (# starts a comment)
<model>/<engine>/v<X.Y.Z>/*.patch      git diffs against the untouched upstream files
<model>/<engine>/v<X.Y.Z>/README.md    one card per patch
<model>/<engine>/v<X.Y.Z>/test-only/   patches used only for testing, not in series
scripts/apply.sh                       apply a version folder to a tree
scripts/check-upstream.sh              apply a version folder to the upstream tag's files
```

## Applying

The patches are `git diff` files with paths relative to the package root (`a/vllm/...`,
`a/lmcache/...`). That root is the top of a source checkout at the tag in `UPSTREAM`, and
also the `site-packages` directory of the installed PyPI wheel.

```bash
scripts/apply.sh qwen3.8-27b/vllm/v0.33.0 /path/to/site-packages --check   # CLEAN / APPLIED / CONFLICT
scripts/apply.sh qwen3.8-27b/vllm/v0.33.0 /path/to/site-packages           # apply the whole series
scripts/check-upstream.sh qwen3.8-27b/vllm/v0.33.0                          # test against the upstream tag
```

`apply.sh` first tries the whole series on a scratch copy and touches the tree only when
every patch applies with zero fuzz. It needs GNU patch. `check-upstream.sh` needs curl.

Then set the switches listed in the model README in the engine's environment. The cards
list the log line that confirms each patch is active.

## How patches are checked

- Every series applies with zero fuzz to its upstream tag (`check-upstream.sh`). The model
  README gives the date of the last check.
- Numeric kernel changes are compared through the decode logit distribution against the
  unpatched path, using the run-to-run KL of the unpatched build as the noise floor. We do
  not expect bit-identical tokens across builds or kernels. Greedy outputs are reported
  where we ran them.
- KV cache store changes are checked with GPU checksums of every chunk on store and
  retrieve (for LMCache: `qwen3.8-27b/lmcache/v0.5.6/test-only/kv_probe.patch`).

## Adding a model or a version

1. A new model gets `<model>/README.md` with its scope and test rig (copy the layout of
   [`qwen3.8-27b/README.md`](qwen3.8-27b/README.md)) and a row in the Models table above.
2. A new engine version gets `<model>/<engine>/v<X.Y.Z>/` with `UPSTREAM`, `series` and the
   patch files. Make each patch as a `git diff` against the untouched upstream files of
   that tag, one patch per change.
3. Name a patch `<id>-<short-name>.patch`. The id (E003, E014, …) is our experiment number,
   unique across the repo, and stays the same when the patch is ported to a newer engine
   version.
4. Write a card per patch in the version folder's `README.md` (template below) and add
   the version to the model README's index.
5. Run `scripts/check-upstream.sh <version folder>`; every line must read OK.

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
Claude), then reviewed, tested and measured by us on the rig named in each model README.
The cards name who wrote each one.

Apache License 2.0, the license of vLLM and LMCache. The patches modify files of those
projects; the original code keeps its authors' copyright.
