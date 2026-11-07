# 06 torch.compile and CUDA graphs

Scope: vLLM 0.33.0 with `FULL_AND_PIECEWISE` CUDA graph mode, torch 2.14.0, the AOT
compile cache of 0.33. The E006c card in
[vllm/v0.33.0](../../../qwen3.8-27b/vllm/v0.33.0/README.md) is the worked example.

## Trace time vs run time

- vLLM traces the model once with torch.compile using symbolic shapes at
  `max_num_batched_tokens`, then captures CUDA graphs per batch size from the compiled
  graph.
- Python control flow inside the traced code runs once, at trace time. E006a put
  `if x.shape[0] <= 16` in `UnquantizedLinearMethod.apply`; under tracing the condition was
  decided for the general shape and the compiled graph always called `F.linear`. nsys on
  0.33 showed no `_small_m_splitk_kernel` and `in_proj_ba` still on the cutlass WMMA
  kernel at 1,818 µs per step (E008).
- The same holds for environment variables read in traced code: their value at trace
  time is baked into the graph.
- Check after any such change: the kernel you expect appears in an nsys trace of the
  decode step at the real batch size, and the step time moves. A unit test that calls
  the function eagerly does not prove the compiled graph uses it.

## Custom ops for run-time dispatch

- E006c registers the small-M GEMM as `torch.ops.vllm.small_m_bf16_linear`. The op is
  opaque to the tracer, so the M ≤ 16 decision and the `VLLM_E006_SMALL_M` switch run
  inside the op each time it executes: on eager calls and while the CUDA graph for each
  batch size is captured at start. Each captured graph holds the branch for its own M.
- The op's fake (meta) implementation returns a tensor of shape (M, N) so tracing can
  continue without running the kernel.
- Keep the Triton path's conditions in the op: bf16, 2-D input, no bias, M ≤ 16; the
  rest calls `F.linear`.
- Measured with the op (E011): step 21.15–21.40 → 19.74–19.91 ms; prefill logits
  bitwise identical at 6,639 positions (prefill M is far above 16, so the op takes the
  `F.linear` path there); decode-logit KL 4.11e-4.

## Compile cache keys and logit identity

- 0.33's AOT compile cache compares the traced source (`decorators.py:273-289`)
  and recompiles when the code changes. Environment variables are not part of the key:
  one E011 run with E006c's switch off loaded the graph compiled with it on and ran
  19.64 ms per step. That is why E006c reads its switch inside the op at run time.
- In A/B runs give each variant its own `VLLM_CACHE_ROOT`. Reuse a variant's cache root
  when you rerun it.
- Logit identity, measured with `logit_diff.py gen/gcmp`:

| Pair | KL over top-20 | Run |
|---|---|---|
| same process, rerun | 0 | E017 |
| same tree + same cache root, after a restart | 0 (identical files) | E025 |
| different tree + fresh cache root, same code path | 4.48e-4 | E018 |
| same build, one extra concurrent request (our noise floor) | 1.72e-4 | E011 |

- So a cross-build comparison carries about 4.5e-4 KL before any patch changes the math;
  E006c, PCIe IPC and the Triton GDN kernel all landed at 4.1e-4 to 4.7e-4. Tree and
  cache root were not separated in E018.

## Graphs, launch time and profilers

- A captured graph replays with one `cudaGraphLaunch`. Without a profiler that call is
  58 µs at one request and 74 µs at four (E026).
- nsys with `--cuda-graph-trace=node` records every node: on 0.28 it stretched
  `cudaGraphLaunch` to about 3.6 ms and the spec step from 27.6 to 35.7 ms, while the
  no-spec step went only from 14.2 to 14.3 ms (EXP-0). On 0.33 the launch read 954 µs
  under nsys and 871 µs under the torch profiler (E020). Kernel durations stay usable;
  step wall time and idle gaps do not.
- Backends decide graph coverage. On sm86 FlashInfer supports only single-token decode
  graphs for non-causal attention, so the DFlash2 drafter on FlashInfer runs eager;
  `TRITON_ATTN` reports full graph support and the drafter is captured `(FULL)` (E004,
  EXP-1).
- The first request after start includes JIT work: E003's first code repetition took
  35.12 ms per step against 22.5 ms after. Drop the first repetition or warm up.
- A cold start with torch.compile took about 6–8 minutes to a healthy server on our
  0.28 test instance.
