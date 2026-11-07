# Dead-end log

Keep one entry per idea we measured and dropped, or could not run. The full list for
Qwen3.8-27B is in [results](../../../qwen3.8-27b/results/README.md#dead-ends).

## Entry format

```markdown
### E### short name (dead | blocked), YYYY-MM-DD
- Idea: one sentence, with the mechanism it relied on.
- Setup: engine version, switches, prompt types, requests, repetitions.
- Measured: the deciding numbers against the baseline of the same day (ranges, units).
- Why dropped: the number that decided it.
- Not measured: what would have been needed to be sure (a trace, more repetitions).
- Revisit if: the condition that would make it worth another run.
```

Example:

```markdown
### E019 second Triton GDN spec kernel "triton2" (dead), 2026-11-04
- Idea: a Triton GDN spec-decode kernel shaped for 8 tokens per request at TP2.
- Setup: vLLM 0.33.0 production code, one request, greedy, 300 tokens, 2 repetitions.
- Measured: step 19.44-20.08 ms vs 19.35-19.52 ms for the existing Triton GDN kernel;
  KL 4.70e-4 vs the base build.
- Why dropped: slower than the kernel it was meant to replace.
- Not measured: per-kernel time (no nsys trace taken).
- Revisit if: an nsys trace of triton2 explains the slowdown (its cause is unknown).
```

## Why we keep them

- A dead end with numbers stops the next agent from spending a GPU session on the same
  idea; a dead end without numbers does not, because nobody can tell whether it was
  measured well.
- Several of ours were close calls that depend on the rig: Humming was 0–0.3 ms slower
  than Marlin while another job used the CPU (E023); low verify width K=1 is still open
  because 4-request runs spread 8.99–13.08 ms (E021, E025). Write the condition under
  which the verdict could flip.
- A blocked entry names the blocker exactly (E024: Marlin int8 activations need
  symmetric `uint4b8` weights; our checkpoint is asymmetric with zero points).
