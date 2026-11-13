# LMCache 0.5.6

Upstream: `LMCache/LMCache` tag `v0.5.6`. Apply order: [`series`](series).
In our production since 2026-11-04 with [vLLM 0.33.0](../../vllm/v0.33.0/README.md).
Rig and server flags: [model README](../../README.md#test-rig).
Scope: written for Qwen3.8-27B (48 GDN + 16 attention layers) with the DFlash2 drafter on
2× RTX 3090 only; no attempt at generality. See [Scope](../../README.md#scope-one-model-one-rig).

| Patch | Switch | Files |
|---|---|---|
| E014 | `LMCACHE_MEOW_CKPT_K=<int>` in the vLLM process | new `lmcache/integration/vllm/meow_checkpoint.py`, `lmcache/integration/vllm/lmcache_mp_connector.py` |
| E030 | `LMCACHE_MEOW_L2_SYNC=1` in the LMCache server | `lmcache/v1/distributed/l2_adapters/nixl_store_dynamic_l2_adapter.py` |
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
  stores every chunk, as upstream. Role detection still runs in `__init__` with the switch
  unset (corrected 2026-11-09).
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
  it is null.
- Not checked at start: K is set in the vLLM process and `--separate-object-groups` on the
  LMCache server, and neither side reads the other. The only sign is the info log line
  below. Confirm the server flag yourself before setting K. This includes blend mode,
  which cannot use `--separate-object-groups` on 0.5.6 (2026-11-09).
- Exceptions (corrected 2026-11-09): an exception while rewriting block ids on a store
  (`_apply_meow_checkpoint`) logs one warning and stores every chunk. Exceptions in
  `__init__` propagate and stop the connector from starting: an invalid
  `LMCACHE_MEOW_CKPT_K`, or a drafter group with more than one sliding window, which is
  checked even with the switch unset.
- Unused code: `store_block_ids_for_tracker` and the two helpers only it calls
  (`_storable_tokens`, `_slice_group`), about 108 lines in `meow_checkpoint.py`, have no
  caller in the patch or in `test-only/` (2026-11-09).
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

### E030 Keep the nixl_store_dynamic L2 index in step with the files on disk

- Switch: `LMCACHE_MEOW_L2_SYNC=1` in the LMCache server environment. Unset keeps upstream
  behaviour.
- Files: `lmcache/v1/distributed/l2_adapters/nixl_store_dynamic_l2_adapter.py`
  (`_meow_filename_to_key`; `__init__` calls `_meow_rebuild_index_from_disk`; a
  `register_listener` override; `_execute_load_in_loop` hands FileNotFoundError keys to
  `_meow_drop_missing`).
- Problem: the dynamic adapter answers lookups from an in-memory index and never checks
  the file. Two failures follow, and each one forces the other:
  - A file deleted from outside LMCache stays in the index. Every lookup reports a hit,
    every load fails with FileNotFoundError, and every later store of that key is skipped
    because the key is still indexed, so the chunk is recomputed on every request until
    LMCache's own eviction happens to drop the key. On our rig an external job deleted
    files older than 3 days. Over 9 days the index held 31,663 objects (757 GB) against
    23,324 files (556 GB) on disk. 10,232 load failures came from 172 keys, the first
    chunks of shared system prompts, which are stored first and so reach 3 days first.
    The first request of each agent session recomputed about 30k tokens (TTFT 16-27 s
    instead of about 1 s).
  - After a restart the index starts empty. A file is adopted only when a lookup asks for
    its key, so files that are never asked for again are neither counted against
    `max_capacity_gb` nor evicted, and they pile up across restarts. That is why the
    external deletion job existed.
- Change:
  1. At start, scan the (flat) storage directory, map each `.bin` name back to its
     ObjectKey (kept only when the key maps back to the same name), and index the files
     oldest mtime first, with the adapter's byte count, the base usage count and the
     per-salt bytes. When a listener (the eviction policy) registers, it gets one
     `on_l2_keys_stored` call per key in that order, so the oldest file is the first
     eviction candidate. Same pattern as upstream `RawBlockL2Adapter.register_listener`.
  2. A load that fails with FileNotFoundError drops the key from the index, subtracts its
     bytes and notifies the listeners, so the next store writes the file again.
- Keep when porting: flat layout only (with `shard_dirs` the rebuild is skipped with a
  warning); `.tmp.*` files are ignored; names that do not map back are counted and
  skipped; a failing rebuild logs the exception and leaves the upstream empty index; only
  FileNotFoundError drops a key, other load errors behave as upstream.
- Signs: the start log has `LMCACHE_MEOW_L2_SYNC: indexed N files`; `GET /status` L2
  `stored_object_count` matches the `.bin` count on disk; a deleted file gives one
  `Dynamic NIXL load failed ... FileNotFoundError` per key followed by `dropped N index
  entries whose file is gone`, not a repeat on every request.
- Tested: adapter test with real NIXL POSIX and CPU memory in a scratch directory. Switch
  off reproduces the upstream behaviour (10 checks); switch on passes 15 checks: external
  delete drops the key, the next store rewrites the file and a reload returns the new
  bytes, a restart rebuilds the index and byte counts, the oldest file is fed to the
  policy first, data reads back after restart, and an LMCache delete removes a rebuilt
  file. Read-only scan of our live directory: 23,324 files, 0 unparsed, 0.16 s. In
  production since 2026-11-13: the start log indexed 23,324 files (556 GB) and the index
  count matched the files on disk.

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
