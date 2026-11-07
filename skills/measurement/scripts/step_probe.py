#!/usr/bin/env python3
"""Decode step probe for a vLLM server with speculative decoding.

Sends streaming chat requests and reads the server's /metrics counters before and after
each batch:

  step_ms_wall = decode wall time / delta vllm:spec_decode_num_drafts_total
                 (one draft = one target verify step; at --conc > 1 the counter sums over
                 requests, so the value is per request-step)
  tok_per_step = (delta accepted + delta drafts) / delta drafts
  server_itl_ms = delta inter_token_latency sum / count (the engine's own ITL)

Without speculative decoding the drafts counter stays 0; read server_itl_ms then.

Our measurement default (see ../SKILL.md): --conc 1 --max-tokens 300 --reps 2 on the code,
prose and mixed prompts, --sampling greedy. Requests set ignore_eos and
chat_template_kwargs.enable_thinking=false.

  python3 step_probe.py --base-url http://HOST:PORT --model NAME \\
      --prompts step_prompts.example.json --types code,prose,mixed --out run.json

Needs the httpx package. Sanitized copy of the probe we used; the computation is unchanged.
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time

METRIC_KEYS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "itl_sum": "vllm:inter_token_latency_seconds_sum",
    "itl_count": "vllm:inter_token_latency_seconds_count",
    "iter_count": "vllm:iteration_tokens_total_count",
}

SAMPLING = "greedy"  # greedy: temperature 0; prod: omit temperature -> server's sampling defaults


def need_httpx():
    try:
        import httpx  # noqa: F401
    except ImportError:
        sys.exit("step_probe.py needs the httpx package (pip install httpx)")
    return httpx


def scrape(base):
    httpx = need_httpx()
    txt = httpx.get(f"{base}/metrics", timeout=10).text
    out = {}
    for k, name in METRIC_KEYS.items():
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.e+]+)$", txt, re.M)
        out[k] = float(m.group(1)) if m else float("nan")
    return out


async def one(client, base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if SAMPLING == "greedy":
        body["temperature"] = 0.0
    t0 = time.perf_counter()
    stamps = []
    usage = None
    parts = []
    async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                stamps.append(time.perf_counter())
                parts.append(ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning_content") or "")
    return {"t0": t0, "stamps": stamps, "usage": usage, "text": "".join(parts)}


async def run_batch(base, model, prompt, max_tokens, conc):
    httpx = need_httpx()
    async with httpx.AsyncClient(timeout=600) as client:
        return await asyncio.gather(*[one(client, base, model, prompt, max_tokens) for _ in range(conc)])


def main():
    ap = argparse.ArgumentParser(description="Decode step time from vLLM spec-decode counters.")
    ap.add_argument("--base-url", required=True, help="server root, e.g. http://HOST:PORT (no trailing /v1)")
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--prompts", required=True, help="JSON object {prompt type: prompt text}")
    ap.add_argument("--types", default="", help="comma list of prompt types; default: all in --prompts")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--conc", type=int, default=1, help="concurrent identical requests per batch")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="", help="optional JSON output file")
    ap.add_argument("--sampling", choices=["greedy", "prod"], default="greedy")
    a = ap.parse_args()
    global SAMPLING
    SAMPLING = a.sampling
    base = a.base_url.rstrip("/")
    prompts = json.load(open(a.prompts, encoding="utf-8"))
    types = a.types.split(",") if a.types else list(prompts)
    rows = []
    for t in types:
        for rep in range(a.reps):
            m0 = scrape(base)
            res = asyncio.run(run_batch(base, a.model, prompts[t], a.max_tokens, a.conc))
            m1 = scrape(base)
            d = {k: m1[k] - m0[k] for k in m0}
            out_tok = sum((r["usage"] or {}).get("completion_tokens", 0) for r in res)
            ttft = statistics.median(r["stamps"][0] - r["t0"] for r in res if r["stamps"])
            dec_start = min(r["stamps"][0] for r in res if r["stamps"])
            dec_end = max(r["stamps"][-1] for r in res if r["stamps"])
            dec_wall = dec_end - dec_start
            chunks = [len(r["stamps"]) for r in res]
            gaps = []
            for r in res:
                s = r["stamps"]
                gaps += [s[i + 1] - s[i] for i in range(len(s) - 1)]
            steps = d["drafts"]
            row = {
                "type": t, "rep": rep, "conc": a.conc, "out_tokens": out_tok,
                "ttft_s": round(ttft, 3), "decode_wall_s": round(dec_wall, 3),
                "steps": steps, "accepted": d["accepted"],
                "tok_per_step": round((d["accepted"] + steps) / steps, 3) if steps else None,
                "accept_rate": round(d["accepted"] / d["draft_tokens"], 3) if d["draft_tokens"] else None,
                "step_ms_wall": round(dec_wall / steps * 1000, 2) if steps else None,
                "server_itl_ms": round(d["itl_sum"] / d["itl_count"] * 1000, 2) if d["itl_count"] else None,
                "chunk_gap_p50_ms": round(statistics.median(gaps) * 1000, 2) if gaps else None,
                "chunk_gap_p90_ms": round(sorted(gaps)[int(len(gaps) * 0.9)] * 1000, 2) if gaps else None,
                "client_tok_s": round(out_tok / dec_wall, 1) if dec_wall else None,
                "chunks": chunks,
                "texts": [r["text"] for r in res],
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "texts"}, ensure_ascii=False), flush=True)
    if a.out:
        json.dump({"args": vars(a), "rows": rows, "ts": time.strftime("%F %T")},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
