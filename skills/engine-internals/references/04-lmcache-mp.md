# 04 LMCache multiprocess server

Scope: LMCache 0.5.6 built from the PyPI sdist against torch 2.14, multiprocess server,
`LMCacheMPConnector` with `kv_role` `kv_both`, vLLM 0.33.0 with E005 (group size 16).
Server flags: `--chunk-size 1600 --engine-type default --separate-object-groups`. Patch
cards: [lmcache/v0.5.6](../../../qwen3.8-27b/lmcache/v0.5.6/README.md).

## Pieces and wiring

- The LMCache server is its own process. It keeps chunks in L1 (pinned host RAM) and L2
  (an NVMe directory; our L2 adapter needs the `nixl` package). vLLM loads the connector
  in its scheduler and in each worker.
- vLLM 0.33's `LMCacheMPConnector` name loads LMCache's own implementation
  (`lmcache/integration/vllm/lmcache_mp_connector.py`) when the package imports, and falls
  back to vLLM's bundled copy otherwise (0.33 `lmcache_mp_connector.py:1185-1215`). The
  bundled copy refuses hybrid models (`:70-83`). LMCache 0.5.6's connector supports
  hybrid GDN models and requires mamba cache mode `align` and
  `max_num_batched_tokens >= block_size`.
- The PyPI 0.5.6 wheel's `lmcache.cuda_ops` is built against torch 2.13; under torch 2.14
  it fails to import and LMCache falls back to plain torch copies. We build from the
  sdist; the card has the error text.
- The object key includes the model name, so two models on one server do not read each
  other's objects (E015). A change in KV layout (group size, attention backend, KV dtype)
  needs a fresh L2 directory.
- The server's start log lists the kernel groups it serves. Ours: GDN `hs=560` × 3
  groups of 16 layers, attention 16 layers `kv=2 hs=512`, drafter 5 layers `nh=4 hs=256`.

## Objects per chunk

One 1600-token chunk on one TP rank, bytes = 1600 × heads × head size × (K and V) ×
layers × 1 B (all groups store one-byte elements on our setup):

| Object group | Bytes per chunk per rank | Arithmetic |
|---|---|---|
| attention | 26,214,400 | 16 layers × 1600 × 512 × 2 |
| GDN | 43,008,000 | 48 layers × 1600 rows × 560 B (the GDN state laid out as 1600 rows) |
| drafter | 8,192,000 | 5 layers × 1600 × 512 × 2 |
| all, one object without `--separate-object-groups` | 77,414,400 | sum |

- LMCache 0.5.6 stores only the real GDN state (888,832 B per layer at TP2, padded to
  1600 × 560 = 896,000 B), not the full 1,638,400 B page. Our earlier padding-trim patch
  for 0.5.4 is not needed on 0.5.6 (E010).
- `--separate-object-groups` writes the three groups as three objects. A server can then
  skip a group for a chunk, which E014 relies on.
- Only full chunks are stored; the tail of a prompt shorter than 1600 tokens is computed
  again on the next turn.

## Store, retrieve and checkpoints

- Store: after a prefill, each full chunk's attention, GDN and drafter objects go to L1
  and on to L2.
- Retrieve: the scheduler-side connector looks up the prompt's chunk hashes; the hit is a
  multiple of 1600 tokens. GDN state is needed only at the chunk the request resumes from;
  attention KV is needed for every chunk up to it.
- E014 checkpoints (K=4): GDN objects at chunks with `(i+1) % 4 == 0` and at the last full
  prompt chunk, drafter objects for the 2 chunks up to each of those, attention for every
  chunk. The connector sets the other GDN and drafter block ids to the null block 0; the
  server skips an object group whose ids are all null (`all_null_chunk_masks`), and a
  retrieve resumes from the last chunk that has GDN state.
- Engine modes: 0.5.6 cannot combine blend mode with `--separate-object-groups` on this
  model (two attention-type groups plus one recurrent group raise a RuntimeError in
  `read_set.py:38-69`), so checkpointing runs on the default engine.

Measured (E012, E013, E014):

| What | Result |
|---|---|
| 71,899-token prompt, cold | 48.6 s |
| same prompt from L1, blend / separate groups | 3.02 s / 2.15 s (70,400 cached tokens) |
| same prompt, L2 prefetch after a server restart, blend / separate | 1,132.6 ms / 409.8 ms |
| 20,319-token request in production, first / repeat | 11.42 s / 1.07 s (19,200 cached) |
| L2 bytes on our test workload, E014 K=4 vs every chunk | 6.39 vs 11.30 GB (−43%) |
| cross-session first hit when the shared prefix ends off a checkpoint | 8,000 → 6,400 tokens |

- Restart the LMCache server after a vLLM instance stops and before the next one starts:
  on 0.28 the server held 8.9 GB on each of the two GPUs after a spec instance stopped,
  and the next instance did not fit.
- A test instance whose KV layout differs from production must use its own server or L2
  directory; otherwise it writes objects of another layout into the shared L2 namespace.

## Checking correctness

- True-retrieve test (E012): start vLLM with its dev-mode endpoints, call
  `POST /reset_prefix_cache` before every request so `cached_tokens` can only come from
  LMCache, run single long prompts (4,445 / 9,317 / 22,657 / 71,899 tokens) and multi-turn
  sessions sharing a prefix, greedy, 64 output tokens. Pass 1 stores, pass 2 reads L1,
  pass 3 reads L2 after a server restart. Compare every output with a cold computation.
- kv_probe (`test-only/kv_probe.patch` in the LMCache folder) logs a GPU checksum of
  every block a store reads and a retrieve writes, and poisons target blocks before a
  retrieve. Match each retrieve digest with the store digest of the same chunk and kernel
  group; E012 compared 936 digests and E014 932, with 0 differences each time.
- Expect bitwise equal KV, and do not expect bitwise equal text: GDN is not
  batch-invariant, so a request restored from cache can differ from a cold one at the
  token level (LMCache's own docs say the same and validate at score level).
- E014 fails open: on any exception it logs one warning and stores every chunk.
  Equal object counts per group in L2 mean checkpointing is not active.
