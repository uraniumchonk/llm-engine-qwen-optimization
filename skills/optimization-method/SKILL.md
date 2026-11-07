---
name: optimization-method
description: Load when you plan or run an optimization experiment on an inference engine for one model, or study how the patches in this repo were made. The research loop as we ran it for Qwen3.8-27B (baseline, step anatomy, hypothesis card, patch behind an environment switch whose off state is upstream, CPU and unit tests, GPU A/B on a test instance, numeric validation, staged deploy with dry-run, apply and rollback, run record), the rules we kept, the run record format and the dead-end log.
---

# Optimization method

We ran this loop 28 times between 2026-11-03 and 2026-11-04 (EXP-0, EXP-1, E001–E026;
outcomes in [qwen3.8-27b/results](../../qwen3.8-27b/results/README.md)). It took a
speculative decode step on our rig from 27.6 ms on vLLM 0.28.0 to 19.28–19.42 ms on vLLM
0.33.0 with patches, and the KV pool from 394,971 (unpatched 0.33.0) to 488,000 tokens.
Coding agents wrote the patches (each card names which); every GPU measurement ran on
our rig.

## The loop

1. **Baseline.** Measure the production instance without restarting it: step time per
   prompt type, the same with speculation off, KV pool, GPU utilization and clocks
   ([measurement](../measurement/SKILL.md)). EXP-0 found a 27.6 ms spec step against a
   14.2 ms no-spec step and a step time that did not follow acceptance; the 13.4 ms gap
   became the target.
2. **Step anatomy.** Split the step until each item is about 1 ms or smaller: an nsys
   kernel census of spec vs no-spec steps (E002, E008), py-spy on engine core and
   workers, GPU cycle phases (E022), CPU timestamps without a profiler (E026). Check every
   profiler number against an unprofiled step before booking it (E020: 1.61 ms idle
   under nsys, an estimated 0.8–0.95 ms without). Engine layers:
   [engine-internals](../engine-internals/SKILL.md).
3. **Hypothesis card.** One change, its mechanism, the predicted effect with the
   arithmetic labelled "estimate, not measured", the measurement that decides, and what
   would refute it. Example: removing the drafter's host sync was estimated at −4 to
   −6 ms; E003 measured 27.60–31.01 → 22.50–22.75 ms on 0.28.
4. **Patch behind a switch.** One change per patch, a `git diff` against untouched
   upstream files. An environment variable selects the new path; unset means the
   upstream path. Print a confirmation line at start (`E005: VLLM_KV_GROUP_SIZE overrides
   5 -> 16 ...`). Reject invalid values loudly (E005 raises a ValueError) and fall back to
   upstream where that is safe (E014 logs one warning and stores every chunk). Read the
   switch where tracing and caches cannot freeze it (E006c reads it inside a custom op).
5. **CPU and unit tests.** Test the arithmetic on the CPU (E001 checked all 256 fp8
   encodings), the kernel in isolation on the GPU (E003 byte equality with torch's fp8
   cast; E006a error at M=8), connector logic with a CPU test (E014), and that the patch
   applies with zero fuzz.
6. **GPU A/B on a test instance.** Stop production if it shares the GPUs, start a test
   instance per variant with its own compile cache, measure both sides the same day with
   the same command, restore production, check for orphan workers.
7. **Numeric validation by class.** Greedy identity for drafter-only and layout changes,
   decode-logit KL against a same-build noise floor for target-math changes, bitwise
   checksums for KV store and retrieve ([validation](../measurement/references/validation.md)).
   A speed number without its correctness check does not go to production.
8. **Staged deploy.** Back up the full production state and write a restore script whose
   default is a dry-run; run the dry-run (E009: 0 drifted files). Rehearse on a test port
   with production settings and production sampling (E013: 20.09–20.25 ms per step;
   retrieve outputs checked against cold compute). Cut over with a script that also
   defaults to dry-run and backs up every file it replaces; after apply, check health,
   the KV pool line, the patch confirmation lines and the logs for errors. Keep server
   configs in version control and audit the logs over the first 24 hours.
9. **Run record and outcome.** Write the record (below) the same day, then set the outcome:
   adopted, dead, blocked or open, with the deciding numbers.

Loop back from step 2 or 3 with the next largest item. Our order of gains is the
[adopted path](../../qwen3.8-27b/results/README.md#adopted-path).

## Rules we kept

- Compare two setups on the same day, the same machine and the same command, and change
  one thing between them. EXP-1 changed the attention backend, KV dtype and context
  length at once, so its KV pool drop cannot be assigned to any one of them.
- One patch per change, one commit per patch, named after its experiment id; a patch
  keeps its id when ported to a newer engine version.
- Correctness evidence comes with every speed patch (step 7 of the loop).
- A change of KV layout (group size, attention backend, KV dtype) gets a fresh external
  KV store directory.
- "The patch removes a sync" and "the step got X ms shorter" are two sentences; the
  second needs its own measurement. E006a added a faster kernel and passed its unit test;
  E008 then showed the compiled graph never called it.
- Treat another agent's summary as a lead: read the raw lines it cites before you quote
  it.
- Port to a new engine version by reading the patch's card, re-applying with zero fuzz,
  and re-measuring; carried-over numbers stay labelled with the version they were
  measured on.

## Records

- **Run record**, one file per measured run, written the same day. Fields and an example:
  [references/run-record.md](references/run-record.md).
- **Dead-end log.** A dropped idea stays in the record with its numbers and the reason,
  so nobody repeats it blind: E019's "triton2" kernel (slower), E023's TritonW4A16 (2.4×
  longer step), E018's 32k vocabulary subset (fewer code tokens per step). Format:
  [references/dead-ends.md](references/dead-ends.md).
- **Corrections, not deletions.** When a number turns out wrong, keep it and add a dated
  correction with the source. Examples: the first E006 measurement (log overwritten,
  compile state unknown) was withdrawn by E011; the cross-restart logit floor of E018
  (4.48e-4) was corrected by E025 (same tree and cache root give 0).
- **Three sentence kinds.** Measured: what, setup, n, date, range. Estimate: labelled
  "estimate, not measured", with the arithmetic and its inputs. Decision: what we chose and
  the measured reason. Ranges over means, units on every number.
