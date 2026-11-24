# Guide: Inkling 2 Mini on an Intel Arc Pro B70 with vLLM (our setup)

This guide reproduces the server we run: **~90 t/s decode on code, ~71 t/s on prose, ~2,500 t/s prefill,
262K context**, on one 32 GB Intel Arc Pro B70, usable from opencode and any OpenAI-compatible client.

## Credits

All the heavy lifting is the work of **[@Danmoreng](https://github.com/Danmoreng)**:
the [Inkling B70 vLLM recipe](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm) (native SYCL kernels,
EXL3 experts, DSpark integration) and the [Qwen B70 vLLM build](https://github.com/Danmoreng/intel-b70-qwen38-vllm)
it builds on. It also uses [Trellis Serve](https://github.com/sybil-solutions/trellis-serve) kernels and the
[grant-ai EXL3 checkpoint](https://huggingface.co/grant-ai/Inkling-2-Mini-EXL3-4bpw) of
[Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

What this guide, written by [Hoopytech](https://hoopytech.com), adds on top:

- Two small **compatibility patches** for building from source on a different host (`patches/vllm-recipe/`):
  C++20 for the PyTorch 2.15 headers, an optional daily stop and a configurable bind address.
- A **tuned profile**, measured on our card: the recipe's opt-in `optimized` prefill profile, and 3 DSpark
  proposals instead of 7.
- Numbered scripts, a systemd unit, an opencode provider and a benchmark.

Everything is pinned to the commits we validated (see `scripts/vllm/env.sh`).

## What you need

| | Tested |
|---|---|
| GPU | Intel Arc Pro B70 32 GB (Battlemage BMG-G31), `xe` kernel driver, power cap 230 W |
| OS | Linux x86-64 (Ubuntu, kernel 7.0), Docker with overlay2, Python ≥ 3.12, git, curl |
| Disk | **~60 GB free on the Docker disk** (runtime image 23.6 GB + builder + build cache), plus 16 GB for weights inside the recipe checkout |
| Time | ~1–3 h unattended (image build), ~15 min for the rest |

The GPU must be free during kernel checks and serving. Stop any other model server on it first.

## Steps

Run everything from a clone of this repository. Each step is a script you can read before running.

> **Status:** we ran these steps by hand on our host. The numbered scripts wrap exactly those commands, and both
> patches were checked to apply cleanly on fresh clones of the pinned commits, but the scripts themselves have not
> yet been run in sequence on a clean machine. If a step fails, please open an issue with its output.

```bash
source scripts/vllm/env.sh   # check INKLING_DIR and INKLING2_RENDER_NODE (ls -l /dev/dri/by-path/)

scripts/vllm/01-clone.sh                 # recipe + build sources at pinned commits, applies our 2 patches
scripts/vllm/02-build-images.sh          # vLLM 0.33 XPU runtime + oneAPI builder (long, no GPU needed)
scripts/vllm/03-download-weights.sh      # EXL3 4bpw checkpoint, 41 files, sha256-verified (~16 GiB)
scripts/vllm/04-build-and-check-kernels.sh   # compiles kernels and runs the numerical checks on the GPU
scripts/vllm/serve.sh                    # foreground; first start takes ~1.5 min
```

The checks in step 4 must all print `PASS`/`Passed`. On our card the errors were ≤ 1.3e-4 (attention, MoE,
prefill) and 1.4e-5 (DSpark taps). Do not serve a build that fails them.

Test from another terminal:

```bash
curl -s http://localhost:8093/health
python3 bench/vllm_bench.py http://localhost:8093 inkling2-mini-exl3-b70 mybox
```

### Keep it running

[`deploy/systemd/inkling-vllm.service`](../deploy/systemd/inkling-vllm.service). Adjust `User`, `INKLING_DIR`
and the script path, then `sudo systemctl enable --now inkling-vllm`.

### Use it from opencode

Add the provider from [`deploy/opencode/opencode.json`](../deploy/opencode/opencode.json) with
`baseURL: http://YOUR_HOST:8093/v1` and the model id **`inkling2-mini-exl3-b70`**, `limit.context: 262144`.

## Expected results (our card)

| | Value |
|---|---|
| Decode, code prompt | 90–97 t/s (DSpark acceptance varies at temperature 1.0) |
| Decode, prose prompt | ~71 t/s |
| Prefill, cold 12K / 48K | ~2,470 / ~2,620 t/s |
| Context | 262,144 tokens (input + output), FP8 KV |
| Retrieval | needle at 150K correct |

Full measurements, including what did not help: [`results/2026-11-23-vllm-exl3.md`](../results/2026-11-23-vllm-exl3.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Kernel build fails with `error: expected ';' after top level declarator` in `TensorBase.h` | PyTorch 2.15 headers need C++20. Make sure `01-clone.sh` applied both patches. |
| `Der SYCL-Test braucht bis zu … Minuten vor 22:45 Europe/Berlin` | The recipe's build/check scripts refuse to start close to 22:45 Berlin time (the author's host schedule). Run them earlier in the day. Serving is not affected (`INKLING2_NO_DEADLINE=1`). |
| `test -c "$render_node"` fails | Wrong render node: set `INKLING2_RENDER_NODE` (e.g. `/dev/dri/renderD128`). |
| Server only reachable from localhost | Set `INKLING2_VLLM_HOST` (default in `env.sh` is `0.0.0.0`). |
| Empty answers with small `max_tokens` | The model reasons even with `reasoning_effort: "none"`: leave room in `max_tokens`. |
| A second client waits | The recipe serves **one sequence at a time**. Concurrent requests queue. |
| `ERROR … model type inkling2_loop … Transformers does not recognize` in the log | Harmless; the recipe's plugin provides the architecture. |
| Weights symlinked to another disk are not found | The checkout is bind-mounted into the container: keep real files under `models/exl3`. |

## About Hoopytech

<a href="https://hoopytech.com"><b>Hoopytech</b></a> is a boutique AI consultancy. We build AI solutions (LLMs,
agents, RAG, MLOps), embed senior AI specialists and fractional AI leadership in client teams, and help with
EU AI Act compliance. Running capable open models on your own hardware, like this setup, is part of that work.

If you want a private LLM deployment on your infrastructure, or a hand tuning one: **[hoopytech.com](https://hoopytech.com)**.
