# Validation methods

Pick the method by what the patch touches (rule 7 in [../SKILL.md](../SKILL.md)).

## Greedy identity (drafter-only and layout-only changes)

- Why it works: with speculative decoding the target verifies every token, so a change
  that only alters what the drafter proposes leaves greedy output unchanged in theory. In
  practice the verify batch shape follows acceptance, and a different shape can change
  target floating-point results for a few tokens.
- How: keep the generated texts of `step_probe.py` (`--out` stores them) for baseline and
  patch at temperature 0, same prompts and `--max-tokens`, and count identical texts.
  Report "k/N identical" and acceptance or tokens per step next to it.
- What we saw: E003 (drafter attention backend) 6/6 on 0.28 and 0.33; E005 (KV grouping)
  6/6. PCIe IPC all-reduce changes the target's reduction order: 0/6, which moved it to
  the KL method. LMCache retrieve with speculation on: 21/22 identical to cold compute,
  22/22 with speculation off (E012).
- A text mismatch alone does not fail a target-math patch; switch to the KL method.

## Logit KL against a noise floor (target-math changes, engine upgrades)

- Run `logit_diff.py gen` on baseline and patched instances with the same prompts and
  `--max-tokens`; run it a third time on the baseline while one extra request runs
  concurrently. `gcmp` compares token by token up to and including the first generated
  token that differs, because after a divergence the two contexts differ.
- Metric: KL(P_a || P_b) on a's top-20 tokens, both sides renormalized on that set, b's
  missing tokens set to b's 20th logprob; mean over compared positions.
- Decide by comparing the patch's KL with the floor and with the cross-build level, and
  write all three with their `compared` counts.

| Pair (2026-11-04, 0.33 production code) | Compared tokens | top-1 agree | KL top-20 |
|---|---|---|---|
| same process, rerun (E017) | 1,995 | 1.0000 | 0 |
| same tree and cache root after restart (E025) | 1,995 | 1.0 | 0 |
| baseline + 1 concurrent request: noise floor (E011) | 1,363 | 0.9978 | 1.72e-4 |
| two variants on the same code path, other tree, fresh cache (E018) | 505 | 0.9881 | 4.48e-4 |
| E006c off vs on, adopted (E011) | 969 | 0.9938 | 4.11e-4 |
| PCIe IPC all-reduce, adopted (E017) | 713 | 0.9916 | 4.64e-4 |
| Triton GDN decode kernel (E017) | 733 | 0.9932 | 4.54e-4 |
| low verify width K=1 (E021, open) | 864 | 0.9942 | 7.38e-4 |

- Reading the table: the adopted patches are 2.4–2.7× the one-extra-request floor and about
  equal to the other-tree pair. That pair is one measurement of 505 tokens.
- `gcmp` reports the mean over positions. The maximum per position is not in this table.
- Engine upgrade from vLLM 0.28.0 to 0.33.0: no KL was recorded; greedy text was 2/6
  identical (E004). The 0.28.0 stack has typev2 speed runs and a corruption soak (360/360
  agent turns clean with LMCache retrieve on, 2026-08-28) in the notes repo. We found no
  output-quality comparison between the two versions (checked 2026-11-09).
- Prefill teacher forcing (`dump`/`cmp`) runs large M only. For E006c it showed bitwise
  identical logits at 6,639 positions while the decode path moved: use `gen`/`gcmp` for
  anything that runs at decode batch sizes.
- This is a relative check, not an absolute KLD against a higher-precision reference
  model.

## Bitwise KV checksums (KV store and retrieve)

- Patch the LMCache server with `qwen3.8-27b/lmcache/v0.5.6/test-only/kv_probe.patch`
  and set `LMCACHE_MEOW_KV_PROBE=1`. It logs one `MEOWPROBE` line per object group per
  chunk with a GPU checksum of every block a store reads and a retrieve writes, fills
  retrieve targets with poison bytes first, and reports `tail_ok` for writes past a view.
  It synchronizes the CUDA stream: tests only.
- Workload: reset vLLM's GPU prefix cache before every request (dev-mode endpoint
  `POST /reset_prefix_cache`) so every cached token comes from LMCache. Run long single
  prompts and multi-turn sessions twice; restart the server between passes to test L2.
- Match each retrieve digest with the store digest of the same rank, chunk and kernel
  group. Pass: zero mismatches and zero unmatched retrieves.
- What we saw: E012 936 digests (speculation on) and 780 (off), 0 differ, GDN
  `tail_ok` 78/78; E014 932 digests, 0 differ, 0 unmatched.
- Bitwise KV does not imply bitwise text: GDN is not batch-invariant, so judge the text
  with the greedy method above.
