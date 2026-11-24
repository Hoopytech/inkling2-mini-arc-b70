"""Compare an OpenAI-compatible server (e.g. the vLLM EXL3 build) against a llama.cpp KL-divergence base file.

The base file is produced by `llama-perplexity --kl-divergence-base BASE -f wiki.test.raw -c 512 --chunks N`
on the reference model (here UD-Q8_K_XL). It stores the exact token ids and, for every scored position, the
reference log-probabilities. We send the same token ids to the server with `prompt_logprobs`, score the same
positions (second half of each chunk, as llama-perplexity does) and report:

  - perplexity of the server model and of the reference, on identical tokens and positions;
  - top-1 agreement: how often the server's most likely token equals the reference's most likely token.

Usage: python3 quality_vs_kld_base.py BASE_FILE MODEL_GGUF URL SERVED_MODEL_NAME
(MODEL_GGUF is only read for tokenizer metadata: BOS id and add_bos flag; needs gguf-py on PYTHONPATH.)
"""
import json
import math
import sys
import urllib.request

import numpy as np
from gguf import GGUFReader

BASE, GGUF_PATH, URL, MODEL = sys.argv[1:5]


def gguf_field(reader, key):
    f = reader.fields[key]
    return f.parts[f.data[0]][0]


reader = GGUFReader(GGUF_PATH)
# llama-perplexity replaces the first token of each chunk with BOS only if the vocab defines one and adds it.
bos = int(gguf_field(reader, "tokenizer.ggml.bos_token_id")) if "tokenizer.ggml.bos_token_id" in reader.fields else None
add_bos = bos is not None and (bool(gguf_field(reader, "tokenizer.ggml.add_bos_token"))
                               if "tokenizer.ggml.add_bos_token" in reader.fields else False)

with open(BASE, "rb") as f:
    assert f.read(8) == b"_logits_", "not a llama-perplexity KLD base file"
    n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
    tokens = np.frombuffer(f.read(int(n_chunk) * int(n_ctx) * 4), dtype=np.int32)
    data_offset = f.tell()
n_ctx, n_vocab, n_chunk = int(n_ctx), int(n_vocab), int(n_chunk)
first = n_ctx // 2
n_scored = n_ctx - 1 - first
nv = 2 * ((n_vocab + 1) // 2) + 4
base = np.memmap(BASE, dtype=np.uint16, mode="r", offset=data_offset, shape=(n_chunk, n_scored, nv))
print(f"n_ctx={n_ctx} n_vocab={n_vocab} chunks={n_chunk} scored/chunk={n_scored} add_bos={add_bos} bos={bos}", flush=True)

nll_srv = nll_ref = 0.0
agree = total = 0
for c in range(n_chunk):
    ids = tokens[c * n_ctx:(c + 1) * n_ctx].tolist()
    if add_bos:
        ids[0] = bos
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "prompt_logprobs": 1, "temperature": 0}
    req = urllib.request.Request(f"{URL.rstrip('/')}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    pl = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["prompt_logprobs"]
    for j in range(first, n_ctx - 1):          # predict token j+1 from prefix [0..j]
        tok = ids[j + 1]
        entry = pl[j + 1]                       # vLLM: entry k = distribution for token k given prefix
        nll_srv -= entry[str(tok)]["logprob"]
        srv_top = int(next(t for t, v in entry.items() if v["rank"] == 1))
        row = base[c, j - first]
        scale, min_lp = np.frombuffer(row[:4].tobytes(), dtype=np.float32)
        vals = row[4:4 + n_vocab]
        nll_ref -= scale * float(vals[tok]) + min_lp
        ref_top = int(np.argmax(vals))
        agree += srv_top == ref_top
        total += 1
    if (c + 1) % 25 == 0:
        print(f"  {c + 1}/{n_chunk} chunks", flush=True)

ppl_srv, ppl_ref = math.exp(nll_srv / total), math.exp(nll_ref / total)
print(json.dumps({"positions": total, "ppl_server": round(ppl_srv, 4), "ppl_reference": round(ppl_ref, 4),
                  "ppl_ratio": round(ppl_srv / ppl_ref, 4), "top1_agreement_pct": round(100 * agree / total, 2)}))
