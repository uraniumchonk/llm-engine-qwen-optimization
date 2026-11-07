# Run record format

One YAML file per measured run, named `<YYYYMMDD>-E###.yaml`. Fields are generalized from
the 21 records we wrote; leave a field out when it does not apply, and write `not_done`
instead of leaving a gap silent.

## Fields

```yaml
run_id: 20261104-E011            # date + experiment id
question: one sentence           # what this run decides
when: 2026-11-04 00:14-00:58 +08 # start-end with time zone
stack: >                         # everything a rerun needs to match
  engine version and the commit of the patched tree; torch, Triton, FlashInfer;
  GPU model, count, tensor parallel size; model, drafter, speculative tokens;
  --kv-cache-dtype, --max-model-len, --gpu-memory-utilization; KV connector on or off
patch: file names and commits; the switch and its values per variant
finding: what led to this run (a trace, a failed earlier run), with its source
unit_tests:
  name_of_test: PASS or FAIL with the numbers (max error, cases)
variants:
  base:    {env: "", log: start log file, kv_pool: tokens, confirm: log line seen}
  patched: {env: "SWITCH=1", log: ..., kv_pool: ..., confirm: ...}
launch_failures_fixed:           # what broke at start and how it was fixed
  - one line each
environment: governor, load average, other jobs running, compile cache root per variant
measurements:
  tool: the exact command line
  step_ms: {variant: {type: [rep1, rep2]}}   # raw values, min-max is derived
  tok_per_step: {variant: {type: value}}
  c4_server_itl_ms: {variant: {type: value}}
  greedy_text_identical: k/N
logits:
  method: tool and mode (gen/gcmp, top-20)
  pair_name: {compared: n, top1_agree: x, kl_top20: x, mean_abs_dlogp_tok: x}
  noise_floor: {compared: n, kl_top20: x}
notes:                           # anomalies, open doubts, dated corrections
  - YYYY-MM-DD correction: what changed and the source
not_done: what was planned and not run
outcome: adopted | dead | blocked | open, with the deciding numbers
rollback: the command or the switch value that restores the previous state
```

- Keep raw files (probe JSON, start logs, traces) next to the record or name their paths.
- Copy numbers from tool output; do not round in the record.
- Write measured values, estimates and decisions in separate fields or sentences.

## Example (E011, shortened)

```yaml
run_id: 20261104-E011
question: does a custom op make the E006a small-M GEMM run inside the compiled graph?
finding: >
  E008 nsys on 0.33 showed no _small_m_splitk_kernel; in_proj_ba still ran the cutlass
  WMMA 16x16 kernel at 1817.99 us per step. The M <= 16 check sat in
  UnquantizedLinearMethod.apply and was fixed at torch.compile trace time.
patch: E006c-small-m-custom-op.patch; switch VLLM_E006_SMALL_M=0 turns it off, read in the op
measurements:
  tool: step_probe.py --types code,prose,mixed --max-tokens 300 --conc 1 --reps 2
  step_ms:
    off: {code: [21.23, 21.16], prose: [21.15, 21.18], mixed: [21.30, 21.40]}
    on:  {code: [19.91, 19.81], prose: [19.74, 19.74], mixed: [19.79, 19.79]}
  c4_server_itl_ms: {on: {prose: 31.21, code: 32.08}}
logits:
  prefill_teacher_forced: 6 texts, 6,639 positions, bitwise identical (prefill M >> 16)
  decode_off_vs_on: {compared: 969, top1_agree: 0.9938, kl_top20: 4.11e-4, mean_abs_dlogp_tok: 8.30e-3}
  noise_floor_on_vs_on_plus_1_concurrent: {compared: 1363, top1_agree: 0.9978, kl_top20: 1.72e-4}
notes:
  - the AOT compile cache keys on source, not on environment variables; one run with the
    switch off loaded the graph compiled with it on (19.64 ms per step); the switch moved
    into the op
  - the earlier E006a/b measurement (20.84-21.02 ms) is withdrawn: its log was
    overwritten and its compile state is unknown
outcome: adopted; -1.3 to -1.6 ms per step at KL 4.11e-4 against a 1.72e-4 floor
rollback: VLLM_E006_SMALL_M=0
```
