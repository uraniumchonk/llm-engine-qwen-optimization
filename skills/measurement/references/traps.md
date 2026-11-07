# Measurement traps we hit

Each entry: what happened, how large it was, what to do. Experiment ids point to
[qwen3.8-27b/results](../../../qwen3.8-27b/results/README.md).

## Numbers that moved without a code change

| Trap | Size | Do |
|---|---|---|
| Other CPU jobs on the machine (E021) | step 19.77 → 22.54 ms (code, one request) | read load average and `top` before each run; rerun when another job overlapped |
| CPU governor `schedutil` vs `performance` (E025) | one request 19.75–19.93 → 19.58–19.71 ms; 4-request code single runs spread 8.99–13.08 ms under `schedutil`, three runs 8.87–8.94 ms under `performance` | record the governor; repeat 4-request runs at least 3 times |
| First repetition after start (E003) | 35.12 ms vs 22.5 ms per step | warm up or drop repetition 0 |
| Control run noise on 0.28 (E003) | two repetitions of the same control differed by 3.3 ms (prose 27.74 vs 31.01) | expect outliers; report min–max, not one value |
| Prompt length (E003) | mixed tokens per step 3.07 at 512 tokens vs 2.49 at 300 | compare only runs with the same `--max-tokens` |
| Engine version changes target numerics (E004) | greedy text 0.33 vs 0.28: 2/6 identical | validate an upgrade with logits (rule 8), not text identity |

## Profilers and tracing

- nsys `--cuda-graph-trace=node`: spec step 27.6 → 35.7 ms, no-spec 14.2 → 14.3 ms
  (0.28); CUDA graph launch 58 µs → 954 µs (0.33). Kernel durations are fine.
- torch profiler with stacks: GPU idle in the prep phase 2.14 ms vs 1.61 ms under nsys vs
  an estimated 0.8–0.95 ms without a profiler (E020). Use its proportions only.
- nsys writes the report only after it receives SIGINT. Outside the capture range there is
  no tracing cost, so time the real step on the same instance before `/start_profile`.
- py-spy with `--threads` or `--idle` on a vLLM worker (about 76 threads) never returned.
  Use `--nonblocking` without those flags, record 3 s pieces under `timeout`, and parse
  its output tolerating broken lines. With `ptrace_scope=1` py-spy needs root.

## Process handling

- `pgrep -f` or `pkill -f` with a pattern from the launch command also matches the shell
  that runs it; one such `pkill` ended the calling shell. Find vLLM processes by PID from
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and check each command line.
- Killing the vLLM main process can leave spawned workers alive, each holding its GPU
  memory (about 20 GB per GPU here). Check `nvidia-smi` after every stop.
- After a spec instance stopped, the LMCache server held 8.9 GB on each GPU and the next
  instance did not start. Restart the KV-cache server between instances.
- Issue stop and start of services as separate commands. One combined stop and start
  command of ours ended with exit 144 (cause not found) although the services came up.

## Compile cache

- 0.33's AOT compile cache keys on source, not on environment variables. A run with a
  switch read at trace time loaded the graph compiled with the other setting (E011).
  Give each A/B variant its own `VLLM_CACHE_ROOT`.
- Same tree and same cache root after a restart give bitwise identical logits (E025); a
  new tree with a fresh cache gave KL 4.48e-4 on the same code path (E018). A logit
  comparison across builds starts at that level.

## Counters

- At several requests `spec_decode_num_drafts` counts each request's draft, so
  `step_ms_wall` is per request-step (8.13 ms for 4 prose requests while the engine step
  was 30.5 ms, E026).
- A test instance without speculation has no drafts counter; use `server_itl_ms`.
