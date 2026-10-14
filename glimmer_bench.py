#!/usr/bin/env python3
# Glimmer throughput smoke: TTFT, per-stream decode tok/s, aggregate tok/s,
# plus /metrics acceptance counters (expected ~idle with speculation off).
# Usage: python3 ~/glimmer_bench.py
# Requires Rapid serving Muse-Glimmer-q4-MLX on 127.0.0.1:8001.
import json, time, urllib.request, concurrent.futures

BASE = "http://127.0.0.1:8001"
MODEL = "Muse-Glimmer-q4-MLX"
KEY = "dummy"
N = 2  # must not exceed rapid's max_concurrent_requests=2 admission ceiling
MAX_TOK = 256

def req(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        return resp.status, resp.read()

def metrics():
    try:
        _, b = req("/metrics")
        txt = b.decode()
        lines = [l for l in txt.splitlines() if "accept" in l.lower()]
        return "\n".join(lines) if lines else txt[:1500]
    except Exception as e:
        return "<metrics fetch failed: %s>" % e

def one_stream(prompt, tag):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOK, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True}}
    data = json.dumps(body).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            if b"[DONE]" in line:
                break
            try:
                p = json.loads(line[5:])
            except Exception:
                continue
            if p.get("usage"):
                usage = p["usage"]
            try:
                delta = p["choices"][0]["delta"].get("content", "")
            except Exception:
                delta = ""
            if delta and first is None:
                first = time.time()
    t1 = time.time()
    comp = (usage or {}).get("completion_tokens")
    return {"tag": tag, "ttft": (first - t0) if first else None, "wall": t1 - t0,
            "completion_tokens": comp,
            "decode_tok_s": (comp / (t1 - first)) if (comp and first) else None}

def phase(name, prompts):
    print("\n== %s ==" % name, flush=True)
    m0 = metrics()
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        res = list(ex.map(lambda pt: one_stream(*pt), prompts))
    wall = time.time() - t0
    m1 = metrics()
    tot = sum(r["completion_tokens"] or 0 for r in res)
    ttfts = [r["ttft"] for r in res if r["ttft"] is not None]
    dec = [r["decode_tok_s"] for r in res if r["decode_tok_s"]]
    for r in res:
        if r["ttft"] is not None:
            dtxt = ("%.1f" % r["decode_tok_s"]) if r["decode_tok_s"] else "n/a (no usage)"
            print("  [%s] TTFT=%.2fs wall=%.2fs tokens=%s decode=%s tok/s"
                  % (r["tag"], r["ttft"], r["wall"], r["completion_tokens"], dtxt))
        else:
            print("  [%s] NO FIRST TOKEN wall=%.2fs" % (r["tag"], r["wall"]))
    if tot and wall:
        print("  aggregate: %d tokens / %.2fs wall = %.1f tok/s" % (tot, wall, tot / wall))
    if ttfts:
        print("  TTFT mean %.2fs" % (sum(ttfts) / len(ttfts)))
    if dec:
        print("  decode mean %.1f tok/s" % (sum(dec) / len(dec)))
    return m0, m1

print("warmup...", flush=True)
w = one_stream("Say hi in five words.", "warmup")
print("warmup TTFT=%.2fs tokens=%s" % (w["ttft"], w["completion_tokens"]))

same = "Explain how a suspension bridge carries its load, in detail."
m0, m1 = phase("same-domain x%d (prefix-cache hot)" % N,
               [(same, "same-%d" % i) for i in range(N)])

mixed_q = [
    "Explain how a suspension bridge carries its load, in detail.",
    "Describe the process of making sourdough bread from starter to bake.",
    "What causes the seasons on Earth? Explain the astronomy.",
    "Summarize the plot of Hamlet in your own words.",
]
phase("mixed-domain x%d" % N, [(mixed_q[i % len(mixed_q)], "mixed-%d" % i) for i in range(N)])

print("\n== /metrics accept counters (before vs after same-domain phase) ==")
print("--- before ---")
print(m0)
print("--- after ---")
print(m1)
print("\nnote: speculation is off, so accept counters are expected to show no activity")
