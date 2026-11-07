# llm-engine-qwen-optimization

Engine patches that make one Qwen model run faster on our hardware, and the method behind
them, written for coding agents to repeat on yours.

![Decode step time per stage, 27.6 to 19.3 ms](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes/raw/main/images/step_time_by_stage_20261107.svg)

## What it did

Qwen3.8-27B with a DFlash2 drafter, vLLM + LMCache, 2× RTX 3090:

- Decode step **27.6 → 19.3 ms**.
- Single-stream decode **+39–42% tok/s** over our previous vLLM 0.28.0 stack.
- KV cache **+23%** (396,364 → 488,000 tokens).
- LMCache disk use **−43%** by storing recurrent state only at checkpoints.

Full numbers and charts: [vllm-hybrid-mamba-notes](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes).

## What's inside

- Patches for vLLM 0.33.0 / 0.28.0 and LMCache 0.5.6, one card per patch.
- Scripts that apply a patch set and check it against the upstream release.
- Skills: apply the patches, port them to your hardware, the optimization method,
  measurement, and engine internals from the HTTP request down to the GPU link.
- Every experiment we ran, dead ends included.

## Built for agents

This repo is meant to be operated by a coding agent. Point yours at
[AGENTS.md](AGENTS.md), for example:

> Read AGENTS.md and port this to my machine.

Your agent will first tell you in a few lines what the repo is, how it fits your
machine and what it plans, and start after you agree. The dense knowledge lives in
`AGENTS.md` and `skills/`; this page is the summary.

Questions about this stack reach down to kernels and the GPU features they run on, where
general knowledge is often wrong. Your agent answers from the measurements here and
says when something was not measured.

## Scope

One model on one rig. The patches make no attempt at generality; other GPUs and models
are untested and we do not support them. The port skill gives an agent the method and
the constants to re-derive. The [model README](qwen3.8-27b/README.md) says exactly what
the patches were made for; further Qwen models get their own folder.

Apache-2.0, the license of vLLM and LMCache.
