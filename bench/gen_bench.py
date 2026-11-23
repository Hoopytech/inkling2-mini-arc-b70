"""End-to-end server benchmark: generation speed (and speculative acceptance) through the OpenAI API.

Usage: python3 bench/gen_bench.py http://HOST:PORT
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PROMPTS = {
    "code": "Write a Python LRUCache class with O(1) get and put using OrderedDict, with docstrings and pytest tests.",
    "prose": "Explain in about 300 words what inflation is, its main causes and how a central bank fights it.",
}

for name, prompt in PROMPTS.items():
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "reasoning_effort": "none",
            "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "cache_prompt": False, "seed": 7}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    tm = r["timings"]
    spec = f"  draft accepted {tm['draft_n_accepted']}/{tm['draft_n']}" if tm.get("draft_n") else ""
    print(f"{name:6} gen={tm['predicted_per_second']:.1f} t/s  tokens={tm['predicted_n']}  wall={time.time() - t:.0f}s{spec}")
