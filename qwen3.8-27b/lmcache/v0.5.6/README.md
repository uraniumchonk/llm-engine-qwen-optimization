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
| E033 | `LMCACHE_MEOW_LOCAL_SKIP=1` in the LMCache server and the vLLM process; `LMCACHE_MEOW_LOOKUP_WAIT_MS=<ms>` in the vLLM process | `lmcache/integration/vllm/lmcache_mp_connector.py`, `lmcache/integration/vllm/vllm_multi_process_adapter.py`, `lmcache/v1/distributed/api.py`, `lmcache/v1/distributed/storage_controllers/prefetch_controller.py`, `lmcache/v1/multiprocess/modules/lookup.py`, `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` |
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

### E033 Do not load or lock the chunks vLLM already holds; do not wait two scheduler steps for a lookup

- Switches: `LMCACHE_MEOW_LOCAL_SKIP=1` in both the LMCache server and the vLLM process
  (local-skip); `LMCACHE_MEOW_LOOKUP_WAIT_MS=<ms>` in the vLLM process (lookup wait, we use 5).
  Unset keeps upstream behaviour on both sides.
- Files: `lmcache/integration/vllm/lmcache_mp_connector.py` (`get_num_new_matched_tokens`
  passes the local hit); `lmcache/integration/vllm/vllm_multi_process_adapter.py`
  (`maybe_submit_lookup_request(local_hit_tokens=)`, the `check_lookup_result` wait loop);
  `lmcache/v1/distributed/api.py` (`PrefetchTaskSpec.skip_load_cols`);
  `lmcache/v1/distributed/storage_controllers/prefetch_controller.py` (`_plan_load`,
  `_release_local_prefix`, `_finish_request`); `lmcache/v1/multiprocess/modules/lookup.py`
  (switch, `meow.local_hit_tokens`, `resolve_prefetched_obj_keys(skip_chunks=)`);
  `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` (failed-retrieve cleanup
  passes `skip_chunks`).
- Problem: vLLM does not schedule a new request until LMCache answers how many chunks it
  holds. To answer, the server prefetches every hit chunk into L1 and read-locks it, however
  much of the prefix vLLM already holds in its own KV cache. Two costs follow:
  - With `--l2-store-policy skip_l1` a chunk that was just stored lives only in L2, so the
    request after a store reads the whole prefix back from disk. On our rig a 10K-token
    prefix meant 433 MB read and a one-step increment of 152-194 ms instead of 72-83 ms.
    Over two days of live traffic vLLM logged 3,007 lookups with a weighted mean of
    91.7 ms.
  - `check_lookup_result` never blocks: the LOOKUP ack, the QUERY_PREFETCH_STATUS send and
    its reply each take one scheduler pass, and the scheduler skips the request while the
    result is unknown. Under load a new request waits two whole engine steps of other
    requests. With 4 agent chains in parallel the logged lookup latency averaged 396 ms.
- Change:
  1. The vLLM connector puts the block-aligned local prefix hit the scheduler passes in
     into `request_configs["meow.local_hit_tokens"]` of the lookup key. ObjectKeys do not
     include `request_configs`, so no object key changes.
  2. Server: `skip_load_cols` = local hit // chunk size. After the first load plan, the
     planned cells of those columns become `local_found`: their L1 read locks and L2 locks
     are returned at once, nothing is reserved in L1 or loaded from L2. Every later plan
     sees them as L1-resident, and `_finish_request` counts them as found.
     `resolve_prefetched_obj_keys` starts at the skipped chunk count, so
     `free_lookup_locks` and failed-retrieve cleanup never return a lock that was not
     taken.
  3. Lookup wait: `check_lookup_result` repeats its pass for up to the budget, waiting on
     the outstanding ack or status reply in between. An answer inside the budget admits
     the request in the same scheduler step; otherwise it stays asynchronous as upstream.
- Keep when porting: the server skips only for lookups that carry
  `meow.local_hit_tokens` (other engines' connectors behave as before); skipped chunks never
  exceed the looked-up chunks; vLLM retrieves from the chunk boundary of its local hit, so a
  retrieve never touches a skipped chunk.
- Signs: with the switch on and the prefix held by vLLM, the request after a store reads
  nothing from disk and the server logs `Prefetch request completed ... (N L1, 0 L2)` in
  about 1 ms; `GET /status` shows `read_locked_count` 0 at rest; no `Bypassing LMCache` or
  load failures in the vLLM log.
- Tested: an offline test that drives the real `PrefetchController` and
  `DefaultPrefetchPolicy` with counting fake L1/L2 tiers (7 cases: hit length equal to the
  upstream path in each pair, every L1 lock returned once or kept for retrieve, no L2 lock
  returned twice). On the test instance: after a 10K-token store the next one-step request
  took 65.8 ms and read 0 MB (upstream 151.9 ms, 433.4 MB). Full LMCache load (6,400 cached)
  and local first half plus LMCache second half (16,000 cached, 18 objects read) gave 32
  greedy tokens identical to a cold recompute in two rounds (KL 0). Four agent chains in
  parallel, 25 turns each from empty caches, upstream / local-skip / local-skip + 5 ms wait:
  logged lookup latency 396.2 / 371.6 / 30.7 ms, TTFT p50 1,300.7 / 1,293.5 / 1,221.8 ms,
  p90 5,504.5 / 5,678.1 / 4,089.2 ms, objects loaded from L2 116 / 58 / 58, L1 use 13.5% /
  6.7% / 6.7%, check turns 13/15 / 12/15 / 13/15 greedy-identical (mean KL 7.4e-4 /
  8.0e-4 / 6.0e-4 against a 9.4e-4 same-prompt concurrency floor), no load failure and no
  lock left. In production since 2026-11-13: live lookups 2.4-3.7 ms; the request after a
  10K-token store took 64.5 ms with 0 MB read.
- Upstream behaviour seen while testing (not changed here): for hybrid models the first
  plan returns the locks of every earlier GDN and drafter cell, so when the L1 reservation
  of the last chunk fails, the replan finds no GDN cell to end on and the hit drops to 0
  instead of falling back to an earlier chunk.

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
