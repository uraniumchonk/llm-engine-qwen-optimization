# typev2: what the suite is, how to rebuild it, how to read it

typev2 is a method: six output shapes sent to one server at one and four concurrent
streams, read as decode tok/s and draft acceptance. Acceptance depends on the workload, so
each site builds its conversation corpus from its own records. Ours holds private
conversations and is not in this repo; this page specifies the rest, and the runner is
[typev2_bench.py](../scripts/typev2_bench.py). Our numbers do not transfer to your workload.

Numbers below: Qwen3.8-27B AWQ-INT4 target, DFlash2 W4A16 GPTQ drafter, 7 speculative
tokens, 2× RTX 3090, vLLM version named per row.

## The six types

Prompt and generated tokens: 2026-11-04, one stream, vLLM 0.33.0 + patches.

| Type | Request | Prompt tokens | Generated tokens |
|---|---|---|---|
| numeric | Count up from a random integer, space-separated, at least 128 numbers, no explanation | 92 | 256 |
| code | One iterative Python Fibonacci function, code only | 90 | 88 |
| prose | About 200 characters of Traditional Chinese on a fixed technical topic | 87 | 164 |
| mixed | Short Chinese explanation, a small Python demo, one closing sentence | 106 | 228 |
| realjob | A real multi-turn conversation ending on a user task | 4,217 | 256 |
| chitchat | A short non-task message to a roleplay persona, with its preceding context | 1,855 | 102 |

The four synthetic types are single-turn and carry a random nonce at the start of the
prompt and a salt. realjob and chitchat are the two corpus types. A level of N streams
sends the first N corpus entries, so one stream is always the first entry: one
conversation, not a sample of the corpus.

## What each type measures

Acceptance is from `/metrics` deltas, 2026-09-11 (run 3 of 3, the run that recorded it),
vLLM 0.28.0, 256 max tokens, temperature 0, 1- and 4-stream rows combined. Tokens per step
is 1 + accepted ÷ drafts.

| Type | Accepted / drafted | Acceptance | Tokens per step |
|---|---|---|---|
| numeric | 1,118 / 1,120 | 99.8% | 7.99 |
| code | 379 / 462 | 82.0% | 6.74 |
| realjob | 890 / 2,275 | 39.1% | 3.74 |
| mixed | 705 / 2,968 | 23.8% | 2.66 |
| prose | 419 / 2,737 | 15.3% | 2.07 |
| chitchat | 252 / 2,051 | 12.3% | 1.86 |

- **numeric is a ceiling test.** Almost every draft is accepted, so its tok/s is 8 tokens
  divided by the step time. Arithmetic for 2026-11-04: 8 ÷ 19.4 ms (inter-token latency
  p50) = 412 tok/s; measured 413.7. It carries the step time and says nothing about
  drafter quality or useful throughput. Quote it as a ceiling only.
- **code** is one short function. The 374.5 tok/s in our table is one request, 88 generated
  tokens, one run (2026-11-04); the three 2026-09-11 runs generated 85–88 tokens at one stream.
- **prose and mixed** are the low-acceptance controls. They show how the stack behaves
  when most drafts are rejected.
- **realjob and chitchat** are the types closest to our real use, and they are fitted to
  us twice over: the prompts come from our records, and the drafter was calibrated on our
  traffic ([Fitted to our records](#fitted-to-our-records)).

The `numeric` prompt in `step_prompts.example.json` is a different prompt: 63.5%
acceptance on 2026-11-03 ([step-anatomy](../../../qwen3.8-27b/results/step-anatomy.md)). A
type name alone does not identify a workload.

## Build your own corpus

**realjob**, from your agent harness's saved sessions:

1. Keep user and assistant messages with non-empty text; for multimodal messages keep the
   text parts.
2. Drop tool messages and assistant turns that contain only tool calls. Merge consecutive
   same-role messages.
3. Trim the end so the conversation stops on a real user task: drop trailing assistant
   turns, then walk back past final user turns that are system-injected, non-task, or
   shorter than 8 characters.
4. Keep conversations with at least 2 user turns, at least 3 messages, and 150–12,000
   characters in total.
5. Sort by message count, descending, and take up to 12. Ours: 9 conversations, 3–23
   messages, 1,314–11,960 characters.

What this leaves out, in our corpus and in yours if you follow these steps: system
prompts, tool schemas, tool results and tool-call output. It measures multi-turn text
conversation with an agent. To measure tool-heavy traffic, keep those messages and label
the result as a different suite.

**chitchat**: every user turn that is a short non-task message, with at least one earlier
turn. Keep the last 16 messages before it as context, trimmed from the front to 4,000
characters. Remove duplicate prompts. Sort by context size, descending. Ours: 6 prompts,
2–16 context messages.

**Corpus file**: a JSON list of objects, each with `messages`, a list of `{"role",
"content"}` that ends on a user message. Other keys are ignored. Corpus order decides what
each level sends.

## Run it

```bash
python3 scripts/typev2_bench.py --base-url http://HOST:PORT --model NAME \
    --corpus realjob.json --chitchat-corpus chitchat.json \
    --metrics-url http://HOST:PORT/metrics --repeats 2 --out run.json
```

Point `--metrics-url` at the vLLM process; a router in front of it may not expose
`/metrics`. Add `--no-anticache` for the identical-prompt setting described below. The
four synthetic prompts in the script are the ones we ran, written in Chinese.

The script is the typev2 mode of our runner with the same decode, aggregate and latency
computations. It adds a `/metrics` snapshot per type and sends `enable_thinking=false`
explicitly; our server already defaulted to that. We tested it against a local stand-in
server (2026-11-09), so its output format and request contents are checked and no GPU
number in this repo comes from this copy.

## Run parameters

| | typev2 | `step_probe.py` |
|---|---|---|
| Prompts | six types above | four types in the example file |
| Max tokens | 256 | 300 |
| Ends at | EOS or max tokens | `ignore_eos` |
| Sampling | temperature 0 | `greedy` or `prod` |
| Streams | 1 and 4 | set with `--conc` |
| Warmup | 4 requests, 8 tokens each | see the script |
| Between rows | 2 s cooldown | see the script |
| Thinking | off | off |

- 0 of 26 typev2 result files we checked record a non-zero temperature. The
  production-sampling tok/s on our notes page come from `step_probe.py` with different
  prompts and lengths. Compare within one tool.
- Each request is streamed with usage reporting. Decode tok/s = generated tokens ÷
  (wall time − time to first token).
- Aggregate tok/s = all generated tokens ÷ the slowest request's wall time; it includes
  prefill.
- Completions end at EOS, so generated tokens differ per build and run. Report them.
  Tokens per step also moves with length
  ([traps](traps.md): mixed 3.07 at 512 tokens vs 2.49 at 300).

## Nonce and the prefix cache

- Synthetic types: a new nonce at the start of every request.
- Corpus types, 2026-09-11 runner: no nonce, so the identical full prompt can hit the
  prefix cache.
- Corpus types, the later runner's default: a nonce at the start of the last user turn, so
  the history before it can hit and the last turn cannot. `--no-anticache` removes it.

The published 2026-11-04 realjob and chitchat rows are `--no-anticache` runs, made to
match the 2026-09-11 runner. Compare corpus rows only between runs with the same setting.

Measured 2026-11-04, vLLM 0.33.0 + patches, one run each, nonce run first:

| Type, streams | Decode tok/s, nonce | No nonce | TTFT p50, nonce | No nonce |
|---|---|---|---|---|
| realjob, 1 | 219.1 | 224.0 | 2.17 s | 0.54 s |
| realjob, 4 | 64.8 | 82.9 | 4.77 s | 1.39 s |
| chitchat, 1 | 81.1 | 89.3 | 0.98 s | 0.16 s |
| chitchat, 4 | 56.6 | 98.2 | 0.99 s | 1.09 s |

typev2 is a decode and acceptance suite. To measure the prefix-cache path, send one fixed
long prompt repeatedly and compare the first request's TTFT with the later ones.

## Fitted to our records

The production drafter is the public bf16 DFlash2 quantized by us to W4A16 with GPTQ on
wide layers and RTN fallback on small layers. Calibration used 400 prompts from 3.5 days
of our own logged traffic (August 2026), mixed 50% agent tasks, 30% roleplay, 20% other,
system prompts removed, truncated to the tail of each conversation. The drafter's weights
were not retrained.

Our typev2 corpus comes from sessions of May 2026. Text overlap with the 400 calibration
prompts, checked 2026-11-09 on the first 200 characters of each user turn after removing
whitespace: 0 of 15 final user turns, 1 of 56 user turns of at least 20 characters. The
fit is to the same user, persona and task mix; the test prompts are not the calibration
prompts.

Acceptance by drafter, 2026-08-21, vLLM 0.27.1, 9 + 6 corpus entries, same runner:

| Type | bf16 | Third-party W4A16 | Third-party W8 (data-free RTN) | Ours, W4A16 GPTQ |
|---|---|---|---|---|
| realjob | 36.5% | 37.5% | 39.5% | 38.0% |
| chitchat | 21.5% | 12.1% | 11.7% | 22.4% |
| numeric | 99.9% | 99.8% | 99.7% | 99.9% |
| code | 85.3% | 86.7% | 86.6% | 84.2% |
| prose | 15.1% | 14.3% | 14.4% | 14.4% |
| mixed | 25.5% | 24.3% | 25.0% | 23.3% |

- The difference between our calibration and the third-party quantized drafters is on
  chitchat (about 10 points on this date); chitchat matched bf16. On realjob all four
  drafters are within 3 points.
- Differences of 1–2 points are within what this corpus size can resolve (9 + 6 entries, 1 run).
- The same GPTQ drafter measured chitchat 12.4% on vLLM 0.28.0 (2026-08-28) and 12.3%
  (2026-09-11). We have not explained the drop from 22.4%.
- Acceptance per type was not recorded on vLLM 0.33.0. Whole-run on 2026-11-04: 3,911 /
  12,397 = 31.5% (all six types, nonce run).

A drafter quantized without your data, or a corpus from someone else's records, gives
different acceptance. Measure yours.

## Report format

Write each row with: type, streams, engine version, date, runs (n), max tokens, sampling,
nonce setting, prompt and generated tokens.

1. Decode tok/s per stream and aggregate tok/s as two separate numbers.
2. Range across runs. Our 2026-09-11 column is 3 runs (about ±3%); our 2026-11-04 column
   is n=1 and says so. Measurement rule 6 asks for at least 2.
3. Acceptance as accepted / drafted, with the counts, from `/metrics` deltas. Snapshot
   around each type for a per-type rate; a whole-run delta mixes all six.
4. Tokens per step: 1 + accepted ÷ drafts.
5. TTFT and inter-token latency percentiles, kept apart from decode tok/s.

## Publishing

- Publish this method and aggregate numbers.
- Keep conversation text, prompts, previews and session identifiers private. Corpus files
  contain the full conversations.
- `typev2_bench.py` result files leave out URLs, file paths and prompt text. Result files
  from other runners can list source session names; our older runner's JSON does. Read
  them before publishing.
- The corpus definition above uses filter patterns you write for your own data. Keep those
  patterns private too if they reveal the content of your conversations.

## Decisions

- 2026-11-09, repo owner: numeric is reported as a ceiling only.
- 2026-11-09, repo owner: realjob and chitchat are labeled as fitted to our own workload,
  and the corpus stays private. The method is public.
