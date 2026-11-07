#!/usr/bin/env python3
"""Relative logit check for patches that change target-model numerics. Standard library only.

gen:  greedy chat generation with top-20 logprobs for every generated token (decode path;
      reaches kernels that only run at small M, which a prefill dump does not)
gcmp: compare two gen files, per prompt, up to and including the first generated-token
      divergence
dump: teacher-force text files through /v1/completions with prompt_logprobs=20,
      max_tokens=1, temperature 0, and save the top-20 logprobs at every position
cmp:  compare two dumps position by position

Printed metrics:
  top1_agree  share of positions whose top-1 token is the same
  kl_top20    KL(P_a || P_b) over a's top-20 tokens, both renormalized on that set; a token
              missing from b's top-20 gets b's 20th logprob; averaged over positions
  mean|dlogp_tok|  (gcmp) mean |logp_a(t) - logp_b(t)| for the generated token t when b's
              top-20 holds it

Use it relatively: compare (patch vs baseline) with (baseline vs baseline under another
batch shape or build). It is not an absolute KLD against a reference-precision model.

  python3 logit_diff.py gen --base-url http://HOST:PORT --model NAME \\
      --prompts logit_prompts.example.json --out base.json
  python3 logit_diff.py gcmp base.json patched.json

Sanitized copy of the tool we used; the comparison math is unchanged.
"""
import argparse
import json
import math
import os
import urllib.request


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def save(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def dump(a):
    base = a.base_url.rstrip("/")
    out = {"model": a.model, "items": []}
    for path in a.corpus:
        name = os.path.basename(path)
        text = open(path, encoding="utf-8").read()[a.offset:a.offset + a.chars]
        r = post(f"{base}/v1/completions", {
            "model": a.model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20,
        })
        pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
        rows = []
        for pos in pl:
            if not pos:
                rows.append(None)
                continue
            rows.append({k: v["logprob"] for k, v in pos.items()})
        out["items"].append({"name": name, "rows": rows, "n": len(rows)})
        print(name, "positions", len(rows), flush=True)
    save(out, a.out)


def gen(a):
    base = a.base_url.rstrip("/")
    prompts = json.load(open(a.prompts, encoding="utf-8"))
    out = {"model": a.model, "items": []}
    for name, prompt in prompts.items():
        r = post(f"{base}/v1/chat/completions", {
            "model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        content = r["choices"][0]["logprobs"]["content"]
        rows = [{"tok": c["token"], "top": {t["token"]: t["logprob"] for t in c["top_logprobs"]}} for c in content]
        out["items"].append({"name": name, "rows": rows})
        print(name, "tokens", len(rows), flush=True)
    save(out, a.out)


def gcmp(a):
    """Decode-path compare: positions up to and including the first generated-token mismatch."""
    A = json.load(open(a.a, encoding="utf-8"))
    B = json.load(open(a.b, encoding="utf-8"))
    T = {"n": 0, "top1": 0, "kl": 0.0, "d": 0.0, "nd": 0}
    for ia, ib in zip(A["items"], B["items"]):
        n = top1 = nd = 0
        kl = d = 0.0
        first_div = None
        for i, (ra, rb) in enumerate(zip(ia["rows"], ib["rows"])):
            ta, tb = ra["top"], rb["top"]
            n += 1
            top1 += max(ta, key=ta.get) == max(tb, key=tb.get)
            floor_b = min(tb.values())
            keys = list(ta)
            la = [ta[t] for t in keys]
            lb = [tb.get(t, floor_b) for t in keys]
            za = math.log(sum(math.exp(v) for v in la))
            zb = math.log(sum(math.exp(v) for v in lb))
            kl += sum(math.exp(x - za) * ((x - za) - (y - zb)) for x, y in zip(la, lb))
            if ra["tok"] in tb:
                d += abs(ta.get(ra["tok"], 0) - tb[ra["tok"]])
                nd += 1
            if ra["tok"] != rb["tok"]:
                first_div = i
                break
        print(f"{ia['name']:9s} compared={n:4d} first_divergence={first_div} top1_agree={top1 / max(n, 1):.4f} "
              f"kl_top20={kl / max(n, 1):.2e} mean|dlogp_tok|={d / max(nd, 1):.2e}")
        T["n"] += n; T["top1"] += top1; T["kl"] += kl; T["d"] += d; T["nd"] += nd
    print(f"ALL       compared={T['n']:4d} top1_agree={T['top1'] / max(T['n'], 1):.4f} "
          f"kl_top20={T['kl'] / max(T['n'], 1):.2e} mean|dlogp_tok|={T['d'] / max(T['nd'], 1):.2e}")


def cmp(a):
    A = json.load(open(a.a, encoding="utf-8"))
    B = json.load(open(a.b, encoding="utf-8"))
    tot = {"n": 0, "top1": 0, "kl": 0.0}
    for ia, ib in zip(A["items"], B["items"]):
        n = top1 = 0
        kl = 0.0
        for ra, rb in zip(ia["rows"], ib["rows"]):
            if not ra or not rb:
                continue
            n += 1
            ta = max(ra, key=ra.get)
            tb = max(rb, key=rb.get)
            top1 += ta == tb
            floor_b = min(rb.values())
            keys = list(ra)
            la = [ra[t] for t in keys]
            lb = [rb.get(t, floor_b) for t in keys]
            za = math.log(sum(math.exp(v) for v in la))
            zb = math.log(sum(math.exp(v) for v in lb))
            for x, y in zip(la, lb):
                kl += math.exp(x - za) * ((x - za) - (y - zb))
        print(f"{ia['name']:9s} n={n:5d} top1_agree={top1 / max(n, 1):.4f} kl_top20={kl / max(n, 1):.6f}")
        tot["n"] += n
        tot["top1"] += top1
        tot["kl"] += kl
    print(f"ALL       n={tot['n']:5d} top1_agree={tot['top1'] / max(tot['n'], 1):.4f} kl_top20={tot['kl'] / max(tot['n'], 1):.6f}")


def main():
    p = argparse.ArgumentParser(description="Relative decode/prefill logit comparison against a vLLM server.")
    s = p.add_subparsers(dest="cmd", required=True)
    d = s.add_parser("dump", help="teacher-forced prefill logprobs over text files")
    d.add_argument("--base-url", required=True, help="server root, e.g. http://HOST:PORT")
    d.add_argument("--model", required=True)
    d.add_argument("--corpus", nargs="+", required=True, help="text files; a slice of each is one item")
    d.add_argument("--offset", type=int, default=0, help="first character of each slice")
    d.add_argument("--chars", type=int, default=2400, help="characters per slice")
    d.add_argument("--out", required=True)
    c = s.add_parser("cmp", help="compare two dump files")
    c.add_argument("a")
    c.add_argument("b")
    g = s.add_parser("gen", help="greedy generation with top-20 logprobs")
    g.add_argument("--base-url", required=True, help="server root, e.g. http://HOST:PORT")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True, help="JSON object {name: prompt}; order is kept")
    g.add_argument("--max-tokens", type=int, default=400)
    g.add_argument("--out", required=True)
    gc = s.add_parser("gcmp", help="compare two gen files up to the first divergence")
    gc.add_argument("a")
    gc.add_argument("b")
    a = p.parse_args()
    {"dump": dump, "cmp": cmp, "gen": gen, "gcmp": gcmp}[a.cmd](a)


if __name__ == "__main__":
    main()
