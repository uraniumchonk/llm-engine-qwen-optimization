# llm-engine-qwen-optimization

Engine patches that make one Qwen model run faster on our hardware, and the method behind
them, written for coding agents to repeat on yours.

![Decode step time per stage, 27.6 to 19.3 ms](https://github.com/uraniumchonk/vllm-hybrid-mamba-notes/raw/main/images/step_time_by_stage_20261107.svg)

## What it did

Qwen3.8-27B with a DFlash2 drafter, vLLM + LMCache, 2× RTX 3090:

- Agent-loop TTFT: prefix-cache hits land on a **64-token grid** instead of the KV block.
  vLLM runs hybrid GDN/Mamba models with large KV blocks (several hundred to a few
  thousand tokens; 1,600 here) and stock prefix hits stop at a block boundary, so every
  agent turn recomputes up to a whole block and TTFT cycles with where the previous prompt
  ended. With E031, 29.5K-token turns that keep the system prompt and tools: **TTFT p50
  796 → 230 ms, p90 1,247 → 538 ms**. It needed a fix to vLLM's copy-on-write of shared
  hybrid pages; before it, mid-block hits read partly zeroed KV.
- Decode step on vLLM 0.33.0, patches alone: **27.10–27.47 → 19.74–19.91 ms**. With the
  CPU governor and PCIe IPC all-reduce as well: 19.28–19.42 ms. Our earlier vLLM 0.28.0
  stack: 27.6–31.0 ms.
- Decode tok/s per stream over that 0.28.0 stack, [typev2 suite](skills/measurement/references/typev2-method.md),
  0.33.0 side one run: **+39–42%** at one stream; at four streams +6–21% for five of the
  six types and +56% for chitchat.
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
