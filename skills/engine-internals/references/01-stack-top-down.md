# 01 Stack top down: one request from HTTP to kernels and back

Scope: vLLM 0.33.0 (V2 model runner, the default for this setup) with the patches of
[qwen3.8-27b/vllm/v0.33.0](../../../qwen3.8-27b/vllm/v0.33.0/README.md), LMCache 0.5.6 as a
KV connector, Qwen3.8-27B + DFlash2 with 7 speculative tokens, tensor parallel 2 on
2× RTX 3090. Flags: [model README](../../../qwen3.8-27b/README.md). File and line
references are to the vLLM 0.33.0 release unless marked 0.28.

## Processes

| Process | Does | Measured on our rig |
|---|---|---|
| API server | OpenAI-compatible HTTP (`/v1/chat/completions`, `/v1/completions`, `/metrics`), chat template and tokenization, detokenization and streaming | 0.28, one request: main thread 99% idle in the event loop (py-spy) |
| Engine core | Scheduler (token budget `--max-num-batched-tokens 2048`, `--max-num-seqs 4`), KV cache manager (1600-token blocks per KV cache group, prefix caching), KV connector scheduler side (LMCache lookups), sends each step's schedule to the workers through a shared-memory broadcast queue | 0.28: 98.8% of samples waiting for workers in `shm_broadcast.acquire_read`; schedule + update under 1%, also at 4 requests |
| Worker, one per GPU | Model runner: inputs, attention metadata, CUDA graph replay of target and drafter, sampling and rejection, KV connector worker side | the critical path (next section) |
| LMCache server | Separate process holding L1 (host RAM) and L2 (disk) chunks; the connector in vLLM sends it store and retrieve requests | not a decode bottleneck at one request (0.28 probe) |

- During decode the two TP workers and the engine core each keep about one CPU thread
  busy on our 6-core / 12-thread CPU. Other CPU jobs on the machine lengthen the step
  (19.9 → 22.5 ms in one run, E021).
- 0.33 enables async scheduling for this setup (no disable warning in the start log): the
  worker starts preparing step n+1 while the GPU still runs step n (E026).
- The output path runs backwards: worker → engine core (sampled token ids) → API server
  (detokenize, server-sent events).

## One decode step in the worker

In the order the TP rank 0 worker runs it (segment names from our E026 timestamps,
medians at one request):

1. Request state update and dispatch (136 + 35 µs), `prepare_inputs` (349 µs), block
   tables (92 µs), GDN state preprocessing for align mode (88 µs). This part overlaps the
   GPU running the previous step.
2. Attention metadata per KV cache group. The target's FlashInfer builder calls
   `seq_lens.cpu()` (`v1/attention/backends/flashinfer.py:1678`), a device-to-host copy that
   waits for the GPU to finish the previous target verify and drafter: 16.4 ms at one
   request, 27.1 ms at four. Most of that is waiting, not work.
3. After the wait, on the critical path: FlashInfer metadata (445 µs), embedding inputs
   (248 µs), model inputs (60 µs), CUDA graph launch call (58 µs); 816 µs in total.
4. GPU: the captured graph runs the 64 target layers and the target lm_head, then the
   rejection sampler compares the 7 draft tokens with the target's choices.
5. Sampling on the CPU side (849 µs), then the DFlash2 propose call (205 µs): the drafter
   graph runs 5 sliding-window layers over 7 masked positions in one forward and picks
   candidates through the full lm_head.
6. Sampled ids go back to the engine core; the next `execute_model` starts.

The drafter's own metadata no longer syncs once it runs on `TRITON_ATTN` (E003): Triton
metadata references persistent GPU buffers and its full CUDA graph replay skips the build.
On FlashInfer the non-causal drafter runs eager and its build reaches the same
`seq_lens.cpu()` block (`flashinfer.py:1669-1679`); that is the sync E003 removes. Details:
[03-spec-decode-dflash.md](03-spec-decode-dflash.md).

## Where the time goes on our rig

Decode, one request, production code (E022 nsys cycle, E026 timestamps; full tables in
[step-anatomy.md](../../../qwen3.8-27b/results/step-anatomy.md)):

| Part | ms per step | Lever we used or tried |
|---|---|---|
| Target 64 layers, GPU | 14.22 (Marlin 9.92) | E006 (`in_proj_ba`), PCIe IPC all-reduce, Triton GDN kernel (E017) |
| Target lm_head, bf16 | 1.48 | none yet |
| Drafter 5 layers + its lm_head read | 1.37 + 1.47 | vocabulary subset (E018, open) |
| Prep phase, GPU waiting on CPU | 1.82 under nsys; estimate, not measured, 0.8–0.95 without a profiler | E026 located it; target backend change is open |
| Verify | 0.21 | — |
| Unprofiled step | 19.28–19.42 (PCIe IPC, CPU governor `performance`, E025) | — |

- Without speculative decoding the 0.33 step is 14.23–14.31 ms (E006); speculation adds
  5.0–5.7 ms per step (arithmetic on the E025 and E006 ranges) for up to 8 tokens.
- Four concurrent requests: the engine step is 30.5 ms on the CPU clock (E026); all Marlin
  calls grow from 10.62 to 17.30 ms per step (E022, NVTX step windows).
- Prefill, for scale: a 71,899-token prompt computed cold took 48.6 s; restored from
  LMCache L1 (70,400 cached tokens) it took 3.02 s (E012).
- At one request the scheduler, API server and LMCache connector do not show up on the
  critical path (0.28 probe, EXP-0).

## Tools per layer

| Layer | Tool | Notes |
|---|---|---|
| Whole step | `/metrics` counters (`vllm:spec_decode_num_drafts_total`, `vllm:inter_token_latency_seconds`) via the measurement skill's `step_probe.py` | no restart, no overhead |
| CPU, Python | `py-spy dump` / `py-spy record --nonblocking` on the engine core and workers | do not pass `--threads` or `--idle`: on a worker with about 76 threads py-spy never returned; record 3 s pieces under `timeout` |
| CPU, exact | timestamps written from inside the model runner (our E026 instrumentation, not in this repo) | the only way we found to time graph launch without inflating it |
| GPU kernels | nsys with `--capture-range=cudaProfilerApi --capture-range-end=repeat`, vLLM `--profiler-config '{"profiler":"cuda"}'`, `/start_profile` and `/stop_profile` | kernel durations are reliable; step wall and idle are not (node tracing) |
| CPU + GPU | torch profiler with stacks | inflates launch and metadata time; use proportions only |

Measurement rules for each tool: [measurement](../../measurement/SKILL.md).
