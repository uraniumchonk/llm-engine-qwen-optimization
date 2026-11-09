---
name: measurement
description: Load before you time a decode step, compare two builds or switches, or validate a patch numerically on vLLM with speculative decoding. Holds our eight measurement rules (step time, ITL, KV pool, greedy vs production sampling, repetitions, profiler effects, validation by patch class, logit KL against a noise floor), the A/B procedure, the traps that moved our numbers, and three scripts - step_probe.py (step time from /metrics), logit_diff.py (relative logit comparison) and typev2_bench.py (six-type workload suite).
---

# Measurement

These rules produced every number in
[qwen3.8-27b/results](../../qwen3.8-27b/results/README.md). They were written for one
model with a speculative drafter on a two-GPU rig; the definitions transfer, our numbers
do not. Write each result as a measured sentence: what, setup, n, date, range.

## The eight rules

1. **Step time.** Decode step = decode wall time ÷ Δ`vllm:spec_decode_num_drafts_total`
   (`step_ms_wall` in `step_probe.py`). At one request it equals the server's mean
   inter-token latency (`server_itl_ms`). With several requests the drafts counter sums
   over requests, so `step_ms_wall` is per request-step; take the engine step from
   `server_itl_ms` or from CPU timestamps. Without speculation only `server_itl_ms` exists.
2. **Default conditions.** `step_probe.py --types code,prose,mixed --max-tokens 300
   --conc 1`, `ignore_eos`, thinking off. `--sampling greedy` (temperature 0) compares step
   times. `--sampling prod` omits the temperature so the server's own sampling defaults
   apply (ours: temperature 0.7, top_p 0.88, top_k 20); use it for real acceptance and
   sampling cost. Keep greedy and prod numbers in separate comparisons.
3. **Profilers change the step.** nsys with `--cuda-graph-trace=node` stretched our spec
   step from 27.6 to 35.7 ms and the CUDA graph launch call from 58 µs to 954 µs. Take
   kernel durations from traces; take step wall time and idle gaps from unprofiled runs.
4. **KV pool.** Read `GPU KV cache size: N tokens` in the start log. Write
   `--kv-cache-dtype`, `--max-model-len` and `--gpu-memory-utilization` next to every pool
   you compare.
5. **No KV connector on the test instance by default.** Our step times are without
   LMCache; at one request the connector was not on the decode critical path. A test
   instance with a different KV layout must not write into a shared L2 store.
6. **Repetitions.** At least 2 repetitions per prompt type per round; label a single one
   "n=1". Report min–max. Two repetitions of our production instance differed by
   0.1–0.7 ms per step. Drop the first repetition after a start (JIT: 35.12 ms vs 22.5 ms).
7. **Validate by patch class.**
   - Drafter-only change: the target decides every emitted token, so report the share of
     greedy outputs identical to the baseline (k/N) and acceptance separately. A
     different verify batch shape can still flip a few tokens.
   - Target-math change (kernels, all-reduce, engine version): logit comparison, rule 8.
   - KV store and retrieve (LMCache): bitwise checksums of every stored and restored
     block (`kv_probe`, test-only patch in the LMCache folder).
8. **Statistical standard for target-math changes.** We do not require identical tokens.
   Run `logit_diff.py gen` (greedy, top-20 logprobs per generated token) on both builds and
   `gcmp` (compares up to the first generated-token divergence). Compare the patch's KL
   with both reference levels: one extra concurrent request on the same build gave KL
   1.72e-4 and mean
   |Δlogprob| 3.37e-3 (2026-11-04). A different tree with a fresh compile cache, same code
   path, gave 4.48e-4; the patches we adopted measured 4.11e-4 to 4.64e-4. The prefill
   `dump`/`cmp` mode does not reach kernels that only run at small M.

## A/B procedure

1. Same day, same machine, same command for both sides. Change one thing per variant;
   two settings changed at once cannot be split afterwards (our EXP-1 changed the
   attention backend, KV dtype and context length together).
2. Before each run read the load average and the CPU governor. Other CPU jobs on the
   machine made our step 19.9 → 22.5 ms; governor `performance` vs `schedutil` moved
   it 19.75–19.93 → 19.58–19.71 ms.
3. Stop the production server if the GPUs are shared, restart the KV-cache server, save
   the list of GPU process PIDs (the `nvidia-smi` query in step 6), start the test
   instance with its own compile cache root (`VLLM_CACHE_ROOT`) per variant, wait for
   health 200.
4. Read the start log: KV pool, the patch's confirmation line, the attention and
   all-reduce backends, the CUDA graph mode. A missing confirmation line means the run
   measures something else.
5. Run `step_probe.py` at one request, then at four; run `logit_diff.py gen` on the same
   instance.
6. Stop the instance and check for orphan workers: after the main process exits, a
   spawned worker can keep its GPU memory. List GPU processes by PID
   (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`). Kill only
   the PIDs this test instance started: those absent from the list saved in step 3, other
   than the KV-cache server. Report every other PID to your human and leave it running.
   Restart the KV-cache server, then bring production back and confirm its health and KV
   pool.
7. Write the run record (fields: [optimization-method](../optimization-method/SKILL.md)).

## Scripts

All three take the server root as `--base-url http://HOST:PORT` and the served model name
as `--model`; none has a default host, port or model. `step_probe.py` needs `httpx`;
`logit_diff.py` and `typev2_bench.py` use only the Python standard library.

```bash
# step time, greedy, one request, 2 repetitions per type
python3 scripts/step_probe.py --base-url http://HOST:PORT --model NAME \
    --prompts scripts/step_prompts.example.json --types code,prose,mixed \
    --max-tokens 300 --conc 1 --reps 2 --out base_c1.json
# four concurrent requests, one type
python3 scripts/step_probe.py --base-url http://HOST:PORT --model NAME \
    --prompts scripts/step_prompts.example.json --types prose --max-tokens 200 --conc 4 --reps 2

# logits: baseline, patched, and the noise floor (baseline again with one extra
# concurrent request running from another shell)
python3 scripts/logit_diff.py gen --base-url http://HOST:PORT --model NAME \
    --prompts scripts/logit_prompts.example.json --out base.json
python3 scripts/logit_diff.py gcmp base.json patched.json
python3 scripts/logit_diff.py gcmp base.json base_busy.json
```

- `step_probe.py` prints one JSON line per repetition: `step_ms_wall`, `server_itl_ms`,
  `tok_per_step`, `accept_rate`, `ttft_s`, `client_tok_s`, chunk gap percentiles.
- `gcmp` prints per prompt `compared`, `first_divergence`, `top1_agree`, `kl_top20`,
  `mean|dlogp_tok|`, then `ALL`. Compare `ALL kl_top20` values, and report `compared`
  with them: a pair that diverges early compares few tokens.
- The example prompt files are English renderings of our prompts. Our runs used the
  originals, written in Chinese; the prose, mixed and two of the logit prompts ask for
  Traditional Chinese output. Acceptance rates and token-level results with these files
  differ from ours.
- `typev2_bench.py` runs the six-type suite behind our tok/s tables. Its four synthetic
  prompts are the originals; realjob and chitchat need a corpus built from your own records
  ([typev2-method](references/typev2-method.md)).

## References

- [references/traps.md](references/traps.md): every trap that changed one of our numbers,
  with the size of the change.
- [references/validation.md](references/validation.md): the three validation methods in
  detail (greedy identity, logit KL, bitwise KV checksums) and the values we saw.
- [references/typev2-method.md](references/typev2-method.md): the six-type suite behind
  our tok/s tables: type definitions, how to build a corpus from your own records, run
  parameters, what each type can and cannot show, report format, what to keep private.
- Where the step's time goes: [step-anatomy.md](../../qwen3.8-27b/results/step-anatomy.md).
- Profiler behaviour per layer: [engine-internals](../engine-internals/SKILL.md),
  references 01 and 06.
