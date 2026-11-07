# LMCache 0.5.6

Upstream: `LMCache/LMCache` tag `v0.5.6`. Apply order: [`series`](series).
In our production since 2026-11-04 with [vLLM 0.33.0](../../vllm/v0.33.0/README.md).
Rig and server flags: [root README](../../README.md#test-rig).
Scope: written for Qwen3.8-27B (48 GDN + 16 attention layers) with the DFlash2 drafter on
2× RTX 3090 only; no attempt at generality. See [Scope](../../README.md#scope-one-model-one-rig).

| Patch | Switch | Files |
|---|---|---|
| E014 | `LMCACHE_MEOW_CKPT_K=<int>` in the vLLM process | new `lmcache/integration/vllm/meow_checkpoint.py`, `lmcache/integration/vllm/lmcache_mp_connector.py` |
| kv_probe (test only, `test-only/`) | `LMCACHE_MEOW_KV_PROBE=1` in the LMCache server | `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` |

Build note: the 0.5.6 PyPI wheel's `lmcache.cuda_ops` is built against torch 2.13. Under
torch 2.14 it fails to import (`undefined symbol
_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`) and LMCache falls back to plain
torch copies. We build 0.5.6 from the PyPI sdist against torch 2.14 + CUDA 13 and apply the
patches to that.

Hybrid GDN state on 0.5.6 needs no patch: upstream stores only the real GDN state per
chunk. On our rig the start log shows KernelGroupInfo GDN `hs=560` ×3, attention `kv=2
hs=512`, drafter `nh=4 hs=256`. 0.5.6 cannot combine blend mode with
`--separate-object-groups`.

---

### E014 Checkpoint-only store of GDN and drafter chunks

- Switch: `LMCACHE_MEOW_CKPT_K=<K>` in the vLLM process environment (we use 4). Unset or 0
  stores every chunk, as upstream.
- Files: new `lmcache/integration/vllm/meow_checkpoint.py` (role detection from vLLM's
  `kv_cache_groups`, the set of chunks to keep, block id rewrite);
  `lmcache/integration/vllm/lmcache_mp_connector.py` (reads the switch and the roles in
  `__init__`; `_apply_meow_checkpoint` runs after `GetStoreMetadata` on both store paths).
- Problem: GDN layers carry a recurrent state and the drafter a sliding window, so a
  retrieve only needs their state at the chunk it resumes from. Upstream stores GDN and
  drafter objects for every chunk next to attention KV.
- Change: attention KV is stored for every chunk. GDN state is stored at chunks where
  `(i+1) % K == 0` and at the last full prompt chunk. Drafter state is stored for the
  `ceil(window / chunk_size)` chunks (2 here) up to each checkpoint. For the other chunks
  the patch sets the GDN and drafter block ids to the null block (0). With
  `--separate-object-groups` the server skips an object group whose block ids are all
  null (`all_null_chunk_masks`), and a retrieve resumes from the last chunk that has GDN
  state. The object format is unchanged; L2 objects written without the patch still read.
- Keep when porting: the LMCache server must run with `--separate-object-groups`. Without
  it GDN and attention share one object, the attention ids keep that object live, and the
  null id is read as block 0, which is wrong data. All GDN engine groups are nulled
  together, because the server drops an object-group chunk only when every kernel group in
  it is null. On any exception the patch logs one warning and stores every chunk.
- Signs it is off or broken: the vLLM log must show
  `LMCACHE_MEOW_CKPT_K=4 roles=['gdn', 'gdn', 'gdn', 'attention', 'drafter']` (our model with
  vLLM E005 at group size 16). Roles all `attention` means role detection failed; it then
  stores everything. Equal object counts per group in L2 mean checkpointing is not active.
- Measured (2026-11-04, K=4): on our test workload L2 holds attention 146, GDN 44, drafter
  82 objects (146 each without the patch); L2 bytes −43% (one 25k-token production request:
  −46%). Second-round `cached_tokens` 3,200 / 8,000 / 22,400 / 70,400 equal the prediction.
  kv_probe made 932 store-vs-retrieve checksum comparisons, 0 differed.
- Cost: when a prefix shared across sessions ends off a checkpoint, a new session hits up
  to K−1 fewer chunks (measured 8,000 → 6,400 tokens).
- Written by: Grok; fail-open handling by Claude.
- Upstream status (2026-11-07): we found no LMCache feature that stores GDN state only at
  checkpoints. A vLLM RFC proposes application-declared prefix checkpoints for hybrid
  GDN/Mamba models on the engine side
  ([discuss.vllm.ai](https://discuss.vllm.ai/t/rfc-pr-stack-application-directed-prefix-checkpoints-for-hybrid-gdn-mamba-up-to-7-6x-speedup-on-l40s/2949)).

### kv_probe (test only)

- File: `test-only/kv_probe.patch`, not in `series`.
- Switch: `LMCACHE_MEOW_KV_PROBE=1` in the LMCache server environment.
- What it does: logs one `MEOWPROBE` line per object group per chunk with a GPU checksum of
  every paged block a store reads and a retrieve writes. Before a retrieve it fills the
  target blocks and the padded tail past each view with poison bytes, so a block that
  still holds the stored bytes cannot pass, and a write past the view shows as
  `tail_ok=False`. It synchronizes the CUDA stream, so it is for tests only.
- Use: run a workload twice, then match each retrieve digest against the store digest of
  the same chunk and kernel group.
- Written by: ported to 0.5.6 by Grok.
