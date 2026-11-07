# 07 Multi-GPU communication

Scope: tensor parallel 2 on GPU0 and GPU1 behind one PCIe switch, vLLM 0.33.0 all-reduce
backends, hidden size 5120 in bf16. Tensor parallel 4 was not measured on 0.33.

## Topology

- `nvidia-smi topo -m`: GPU0–GPU1 `PXB` (one PCIe switch between them), PCIe gen4 x16;
  `nvidia-smi topo -p2p r` reports P2P OK.
- Read both before you compare numbers with ours: an all-reduce over a path through the
  CPU root complex, or without P2P, is a different measurement.

## Where the collectives are

- Each target layer all-reduces twice: after the attention or GDN output projection and
  after the MLP down projection. A decode step does 139 all-reduces: 128 for the target,
  11 for the drafter (E022).
- Message size = tokens × 5120 × 2 B: 81,920 B per call at one request (8 verify tokens),
  327,680 B at four requests. A no-spec decode call moves 10,240 B.
- The target logits are all-gathered after the lm_head: 0.12 ms per step at one request
  on 0.33 (E022).
- On 0.28, spec all-reduce calls took 20.9 µs each (grid of 10 CTAs) against 7.3 µs for
  no-spec calls (grid of 2): 2.85× the time for 8× the bytes (E002).

## Backends and what each costs

| Backend at TP2 | How to select | Per call, 80 KB (p10–p90) | Per step, 1 / 4 requests |
|---|---|---|---|
| custom all-reduce, `cross_device_reduce_1stage` | default | 8.19 µs (8.06–8.35) | 1.34 / 3.32 ms |
| FlashInfer PCIe IPC, `ipc_tp2_remote_push_kernel` | `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | 3.30 µs (3.04–4.86) | 1.05 / 2.62 ms |
| NCCL (PYNCCL) | fallback | not measured at TP2 on 0.33 | — |

Durations are the later-arriving GPU's kernel time from pairing both GPUs' calls by order
(nsys, E022); that removes the wait for the other GPU.

- Two-point fits (80 KB and 320 KB): custom 22.0 GB/s with a 4.5 µs intercept, PCIe IPC
  24.6 GB/s with an intercept near 0.
- GPU skew at one request, medians: custom starts 0.92 µs apart and ends 0.10 µs apart;
  PCIe IPC starts 2.82 µs apart and ends 2.63 µs apart.
- Confirm the backend in the start log: `Using ['FLASHINFER_PCIE_IPC', 'CUSTOM',
  'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'` and `Initialized
  FlashInfer PCIe IPC all-reduce for TP2, hidden_dim=5120, dtype=torch.bfloat16,
  max_tokens=64`. Messages over 64 tokens (prefill) go to the next backend in the list.
- PCIe IPC changes the reduction order: greedy text 0/6 identical to custom (E007),
  decode-logit KL 4.64e-4 (E017). Step without a profiler 19.75–19.93 → 19.32–19.52 ms
  (E017); with the CPU governor `performance` 19.58–19.71 → 19.28–19.42 ms (E025). In our
  production since 2026-11-04.

## More than two GPUs

- vLLM 0.33 disables its custom all-reduce when `world_size > 2` and the GPUs are linked
  only through PCIe, logging `Custom allreduce is disabled because it's not supported on
  more than two PCIe-only GPUs` (`distributed/device_communicators/custom_all_reduce.py:267-273`).
  Tensor parallel 4 on such a rig falls back to NCCL.
- Estimate, not measured on 0.33: on our earlier four-GPU 0.28 setup an NCCL call took
  about 29 µs, at about 140 calls per step, about 4 ms per step (a mix of an older
  measurement and our step estimator).
- The FlashInfer PCIe IPC kernel we saw is named for TP2 (`ipc_tp2_remote_push_kernel`);
  which kernel it uses at TP4, and whether it runs there, we have not checked.
- To measure TP4 the way we measured TP2: nsys on both all-reduce backends, pair calls
  across all four GPUs by order, read `nvidia-smi topo -m` for each new GPU's path, and
  compare per-call times with the 80 KB numbers above.
