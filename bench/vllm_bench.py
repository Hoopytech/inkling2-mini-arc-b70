"""Benchmark an OpenAI-compatible Inkling endpoint: cold prefill and streamed decode.

Usage: [REPS=N] [DECODE_ONLY=1] python3 bench/vllm_bench.py http://HOST:PORT MODEL [label]
Cold prefill uses a unique random salt per request so prefix caching cannot help.
"""
import json
import os
import random
import statistics
import sys
import time
import urllib.request
import uuid

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else ""
URL = f"{BASE}/v1/chat/completions"
WORDS = ("the of and to in is that for it as was with be by on not he this are or his from at which but have an "
         "they you were her she there would their we him been has when who will more no if out so said what up its "
         "about into than them can only other new some could time these two may then do first any my now such like").split()
PROMPTS = {
    "code": "Write a Python LRUCache class with O(1) get and put using OrderedDict, with docstrings and pytest tests.",
    "prose": "Explain in about 300 words what inflation is, its main causes and how a central bank fights it.",
}


def post(body, stream=False, timeout=1800):
    data = json.dumps({"model": MODEL, **body, **({"stream": True, "stream_options": {"include_usage": True}} if stream else {})})
    return urllib.request.urlopen(urllib.request.Request(URL, data=data.encode(), headers={"Content-Type": "application/json"}),
                                  timeout=timeout)


def prefill(n_words):
    rng = random.Random(uuid.uuid4().int)
    text = f"[{uuid.uuid4()}] " + " ".join(rng.choice(WORDS) for _ in range(n_words))
    t = time.time()
    r = json.load(post({"messages": [{"role": "user", "content": text + "\nReply with OK."}], "max_tokens": 1,
                        "reasoning_effort": "none"}))
    return r["usage"]["prompt_tokens"] / (time.time() - t), r["usage"]["prompt_tokens"]


def decode(prompt):
    first = usage = None
    with post({"messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "min_tokens": 600,
               "reasoning_effort": "none", "temperature": float(os.environ.get("TEMP", "1.0")), "top_p": 1.0}, stream=True) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            if first is None and ev.get("choices"):
                d = ev["choices"][0]["delta"]
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    first = time.time()
    return usage["completion_tokens"] / (time.time() - first)


post({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}).read()  # warm-up
REPS = int(os.environ.get("REPS", "2"))
res = {}
for n in (() if os.environ.get("DECODE_ONLY") else (12000, 48000)):
    runs = [prefill(n) for _ in range(2)]
    res[f"prefill_{runs[0][1] // 1000}k"] = statistics.mean(r[0] for r in runs)
for name, p in PROMPTS.items():
    # Discard impossible values: if the request queued behind another client, tokens can arrive in a burst.
    vals = [v for v in (decode(p) for _ in range(REPS)) if v < 1000]
    res[f"decode_{name}"] = statistics.median(vals)
    if len(vals) > 2:
        res[f"decode_{name}_sd"] = statistics.stdev(vals)
    res[f"decode_{name}_n"] = len(vals)
print(json.dumps({"label": LABEL, **{k: round(v, 1) for k, v in res.items()}}))
