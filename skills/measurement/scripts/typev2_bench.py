#!/usr/bin/env python3
"""typev2 output-speed suite for a vLLM server with an OpenAI-compatible chat endpoint.

Sends six output shapes at each concurrency level and records, per type and level, decode
tok/s per stream, aggregate tok/s, TTFT and inter-token latency. With --metrics-url it also
records draft acceptance from the server's /metrics counters.

  numeric, code, prose, mixed   built in, single turn, random nonce and salt per request
  realjob, chitchat             your own conversations, from --corpus and --chitchat-corpus

A level of N streams sends the first N entries of a corpus, so one stream is always the
first entry. Corpus file: a JSON list of objects, each holding "messages", a list of
{"role", "content"} that ends on a user message. Other keys are ignored. Build it from your
own records with the steps in ../references/typev2-method.md, which also explains what each
type can show.

  python3 typev2_bench.py --base-url http://HOST:PORT --model NAME \\
      --corpus realjob.json --chitchat-corpus chitchat.json \\
      --metrics-url http://HOST:PORT/metrics --repeats 2 --out run.json

Point --metrics-url at the vLLM process itself; a router in front of it may not expose
/metrics.

Standard library only. Sanitized copy of the typev2 mode of the runner we used; decode,
aggregate and latency computations are unchanged. Three additions: a /metrics snapshot per
type, an explicit enable_thinking=false unless --thinking is set, and result files that
leave out URLs, file paths and prompt text.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import secrets
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

CORPUS_TYPES = ("realjob", "chitchat")


def _rand_int(lo: int, hi: int) -> int:
    return secrets.randbelow(hi - lo) + lo


# The prompts we ran, unchanged, so that synthetic rows use the same requests as ours.
SYNTHETIC_TYPES: dict[str, Any] = {
    "numeric": lambda nonce, salt: (
        f"[nonce={nonce}] 從 {_rand_int(100, 9999)} 開始用空格連續輸出整數遞增，"
        f"至少輸出 128 個數字。不要解釋。salt={salt}"
    ),
    "code": lambda nonce, salt: (
        f"[nonce={nonce}] 用 Python 寫一個函數，功能是計算 Fibonacci 數列第 N 項，"
        f"使用迭代而非遞迴。請只輸出程式碼，不要解釋。"
        f"函數名稱必須包含 salt={salt[:8]} 作為後綴以避免快取命中。"
    ),
    "prose": lambda nonce, salt: (
        f"[nonce={nonce}] 請用繁體中文寫一篇約 200 字的短文，主題是「PCIe 匯流排技術在 "
        f"AI 推理中的角色」。必須包含 unique 參考標記：ref={salt[:16]}。"
    ),
    "mixed": lambda nonce, salt: (
        f"[nonce={nonce}] 請完成以下任務：\n"
        f"1. 用繁體中文解釋什麼是 MTP（Multi-Token Prediction），約 80 字。\n"
        f"2. 然後用 Python 實作一個簡單的 tokenizer demo（只輸出程式碼部分）。\n"
        f"3. 最後用一句話總結。\n"
        f"unique_id={salt[:12]}"
    ),
}


def pctile(data: list[float], p: float) -> float | None:
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] + c * (s[f + 1] - s[f])
    return s[f]


def fmt(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def post_stream(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    temperature: float,
    thinking: bool,
) -> dict[str, Any]:
    """One streaming request; retries 429 and 503 up to 5 attempts."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    data = json.dumps(body).encode()
    failed = {
        "ttft_s": None, "prompt_tokens": 0, "completion_tokens": 0,
        "decode_tok_s": None, "prefill_tok_s": None, "itl_samples": [],
    }

    for attempt in range(5):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        t_start = time.perf_counter()
        ttft: float | None = None
        itl_samples: list[float] = []
        t_last_chunk: float | None = None
        usage: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content") or ""
                    if piece:
                        t_now = time.perf_counter()
                        if ttft is None:
                            ttft = t_now - t_start
                        elif t_last_chunk is not None:
                            itl_samples.append(t_now - t_last_chunk)
                        t_last_chunk = t_now

            wall = time.perf_counter() - t_start
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            decode_s = (wall - ttft) if (ttft is not None and wall > ttft) else None
            return {
                "wall_s": wall,
                "ttft_s": ttft,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "decode_tok_s": (ct / decode_s) if (decode_s and ct) else None,
                "prefill_tok_s": (pt / ttft) if (ttft and pt) else None,
                "itl_samples": itl_samples,
                "error": None,
            }
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 4:
                time.sleep(0.4 * (attempt + 1))
                continue
            return {**failed, "wall_s": time.perf_counter() - t_start, "error": f"HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001
            return {
                **failed,
                "wall_s": time.perf_counter() - t_start,
                "error": f"{type(e).__name__}: {e}",
            }
    raise AssertionError("unreachable")


def load_corpus(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        c for c in data
        if (c.get("messages") or []) and c["messages"][-1].get("role") == "user"
    ]


def build_messages(
    output_type: str,
    job_id: int,
    corpora: dict[str, list[dict]],
    anticache: bool,
) -> list[dict[str, str]]:
    if output_type in CORPUS_TYPES:
        entries = corpora[output_type]
        msgs = [dict(m) for m in entries[job_id % len(entries)]["messages"]]
        if anticache:  # the history before the last turn can still hit the prefix cache
            msgs[-1]["content"] = f"[nonce={secrets.token_hex(8)}] {msgs[-1]['content']}"
        return msgs
    nonce = secrets.token_hex(8) if anticache else "fixed"
    salt = secrets.token_hex(16) if anticache else "fixed"
    return [{"role": "user", "content": SYNTHETIC_TYPES[output_type](nonce, salt)}]


def run_batch(jobs: list[list[dict[str, str]]], call: Any) -> list[dict[str, Any]]:
    if len(jobs) == 1:
        return [call(jobs[0])]
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        return list(ex.map(call, jobs))


def aggregate_level(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if not r["error"]]
    if not ok:
        return {
            "n_ok": 0,
            "n_err": len(results),
            "errors": sorted({r["error"] for r in results}),
        }
    total_ct = sum(r["completion_tokens"] for r in ok)
    batch_wall = max(r["wall_s"] for r in ok)
    all_itl = [x for r in ok for x in r["itl_samples"]]
    decodes = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
    prefills = [r["prefill_tok_s"] for r in ok if r["prefill_tok_s"]]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    return {
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "completion_tokens": total_ct,
        "batch_wall_s": batch_wall,
        "aggregate_tok_s": total_ct / batch_wall if batch_wall else None,
        "avg_decode_tok_s": statistics.mean(decodes) if decodes else None,
        "avg_prefill_tok_s": statistics.mean(prefills) if prefills else None,
        "ttft_p50_s": pctile(ttfts, 50),
        "ttft_p90_s": pctile(ttfts, 90),
        "ttft_p99_s": pctile(ttfts, 99),
        "itl_p50_ms": (pctile(all_itl, 50) or 0) * 1000,
        "itl_p90_ms": (pctile(all_itl, 90) or 0) * 1000,
        "itl_p99_ms": (pctile(all_itl, 99) or 0) * 1000,
    }


def snap_metrics(metrics_url: str) -> dict[str, Any]:
    text = urllib.request.urlopen(metrics_url, timeout=10).read().decode()
    out: dict[str, Any] = {"accepted": 0.0, "draft": 0.0, "drafts": 0.0, "pos": {}}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if "spec_decode_num_accepted_tokens_per_pos_total" in line:
            m = re.search(r'position="(\d+)"[^\}]*\}\s+([0-9.eE+-]+)', line)
            if m:
                out["pos"][int(m.group(1))] = float(m.group(2))
        elif "spec_decode_num_accepted_tokens_total{" in line:
            out["accepted"] = float(line.rsplit(None, 1)[-1])
        elif "spec_decode_num_draft_tokens_total{" in line:
            out["draft"] = float(line.rsplit(None, 1)[-1])
        elif "spec_decode_num_drafts_total{" in line:
            out["drafts"] = float(line.rsplit(None, 1)[-1])
    return out


def metrics_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    accepted = b["accepted"] - a["accepted"]
    draft = b["draft"] - a["draft"]
    drafts = b["drafts"] - a["drafts"]
    return {
        "accepted": accepted,
        "draft": draft,
        "drafts": drafts,
        "rate": (accepted / draft) if draft else None,
        "tokens_per_step": (1 + accepted / drafts) if drafts else None,
        "pos": {
            k: b["pos"].get(k, 0) - a["pos"].get(k, 0)
            for k in sorted(set(a["pos"]) | set(b["pos"]))
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="typev2 output-speed suite",
        epilog="Definitions and how to read the results: ../references/typev2-method.md",
    )
    ap.add_argument("--base-url", required=True, help="server root, http://HOST:PORT")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--corpus", default="", help="JSON file for realjob")
    ap.add_argument("--chitchat-corpus", default="", help="JSON file for chitchat")
    ap.add_argument(
        "--types",
        default="",
        help="comma-separated; default numeric,code,prose,mixed plus each corpus type with a file",
    )
    ap.add_argument("--levels", default="1,4", help="concurrent streams, comma-separated")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=true")
    ap.add_argument("--warmup", type=int, default=4, help="requests before the first row")
    ap.add_argument("--repeats", type=int, default=1, help="runs per type and level")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds after each batch")
    ap.add_argument("--no-anticache", action="store_true", help="send no nonce or salt")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--metrics-url", default="", help="vLLM /metrics, for draft acceptance")
    ap.add_argument("--out", default="", help="write the result JSON to this file")
    args = ap.parse_args()

    files = {"realjob": args.corpus, "chitchat": args.chitchat_corpus}
    default_types = [*SYNTHETIC_TYPES, *(t for t in CORPUS_TYPES if files[t])]
    types = [t.strip() for t in args.types.split(",") if t.strip()] or default_types
    known = (*SYNTHETIC_TYPES, *CORPUS_TYPES)
    unknown = [t for t in types if t not in known]
    if unknown:
        print(f"unknown types: {unknown}; known: {list(known)}", file=sys.stderr)
        return 2

    corpora: dict[str, list[dict]] = {}
    for t in CORPUS_TYPES:
        if t not in types:
            continue
        if not files[t]:
            print(f"{t} needs its corpus file; see --help", file=sys.stderr)
            return 2
        corpora[t] = load_corpus(files[t])
        if not corpora[t]:
            print(f"{t}: no entry in its corpus file ends on a user message", file=sys.stderr)
            return 2

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    def call(messages: list[dict[str, str]], max_tokens: int = args.max_tokens) -> dict[str, Any]:
        return post_stream(
            url, args.model, messages, max_tokens, args.timeout, args.temperature, args.thinking
        )

    print(f"== typev2: model={args.model} types={types} levels={levels} ==", flush=True)
    for i in range(args.warmup):
        r = call([{"role": "user", "content": f"ping {i}"}], 8)
        if r["error"]:
            print(f"warmup failed: {r['error']}", file=sys.stderr)
            return 1

    use_metrics = bool(args.metrics_url)
    run_start = snap_metrics(args.metrics_url) if use_metrics else None
    rows: list[dict[str, Any]] = []
    per_type: list[dict[str, Any]] = []
    for output_type in types:
        type_start = snap_metrics(args.metrics_url) if use_metrics else None
        for level in levels:
            for rep in range(args.repeats):
                jobs = [
                    build_messages(output_type, j, corpora, not args.no_anticache)
                    for j in range(level)
                ]
                row = {
                    "output_type": output_type,
                    "concurrency": level,
                    "repeat": rep,
                    **aggregate_level(run_batch(jobs, call)),
                }
                rows.append(row)
                print(
                    f"  {output_type:<10} n={level:<3} rep={rep} "
                    f"agg={fmt(row.get('aggregate_tok_s')):>7} tok/s  "
                    f"dec={fmt(row.get('avg_decode_tok_s')):>6}  "
                    f"ct={row.get('completion_tokens', 0):<5} "
                    f"ttft_p50={fmt(row.get('ttft_p50_s'), 3)}s  "
                    f"itl_p50={fmt(row.get('itl_p50_ms'))}ms  "
                    f"err={row['n_err']}",
                    flush=True,
                )
                time.sleep(args.cooldown)
        if use_metrics:
            delta = metrics_delta(type_start, snap_metrics(args.metrics_url))
            per_type.append({"type": output_type, **delta})
            rate = "n/a" if delta["rate"] is None else f"{delta['rate'] * 100:.1f}%"
            print(
                f"  {output_type:<10} accepted/drafted "
                f"{int(delta['accepted'])}/{int(delta['draft'])} = {rate}  "
                f"tokens/step {fmt(delta['tokens_per_step'], 2)}",
                flush=True,
            )

    payload = {
        "meta": {
            "suite": "typev2",
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "types": types,
            "levels": levels,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cooldown_s": args.cooldown,
            "no_anticache": args.no_anticache,
            "corpus_entries": {t: len(c) for t, c in corpora.items()},
        },
        "rows": rows,
        "metrics_per_type": per_type,
        "metrics_delta": (
            metrics_delta(run_start, snap_metrics(args.metrics_url)) if use_metrics else None
        ),
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
