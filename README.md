# Inkling 2 Mini on Intel Arc Pro B70

Running [Inkling 2 Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (26B MoE, 5.4B effective active,
looped layers, NVFP4 experts) on a single **Intel Arc Pro B70 (32 GB)** with vLLM XPU and llama.cpp. The repo
has reproducible scripts, measurements, a profile of where decode time goes, and a SYCL fix.

## Credits first

**The fast vLLM setup is [@Danmoreng](https://github.com/Danmoreng)'s work**: his
[Inkling B70 vLLM recipe](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm) (native SYCL kernels, EXL3
experts, DSpark integration) does the heavy lifting. **[Hoopytech](https://hoopytech.com)** reproduced it on a
second B70, and our own contributions are:

- the llama.cpp investigation on this card (Vulkan vs SYCL, decode profile, context limits),
- two small compatibility patches to build the recipe from source on another host,
- a tuning pass of the recipe's options (+43–46 % prefill, +7–9 % decode on our card),
- a step-by-step guide, scripts and benchmarks.

The SYCL `SCALE` fix for llama.cpp is being upstreamed in
[ggml-org/llama.cpp#32784](https://github.com/ggml-org/llama.cpp/pull/32784) by its author; we found and fixed the
same issue independently, and our variant also covers strided/in-place destinations.

## TL;DR

**Use vLLM, not llama.cpp, on this card.** With the recipe above plus our tuning, it runs **3–6× faster** than
the best llama.cpp configuration we found. **All numbers are single-user** (one request at a time; the vLLM
recipe serves one sequence, concurrent requests queue):

| | llama.cpp Vulkan (UD-Q4_K_XL) | vLLM XPU + EXL3 4bpw + DSpark |
|---|---|---|
| Generation, code prompt | ~20 t/s | **90–97 t/s** |
| Generation, prose prompt | ~20 t/s | **~71 t/s** |
| Prefill | ~600 t/s | **~2,500 t/s** |
| Context | 384K (q8 KV) | 262K (FP8 KV) |
| Quality vs native weights (wikitext-2 PPL, top-1 agreement) | −0.05 %, 97.7 % | +0.24 %, 96.3 % |

Code decodes faster than prose because of **speculative decoding**: the DSpark drafter guesses code tokens
(boilerplate, repeated identifiers) far more often than free prose, so each verification pass yields more
accepted tokens. Without speculation both would decode at the same speed (~60 t/s per the recipe author).

**Step-by-step guide to set it up like ours: [`docs/GUIDE-vllm.md`](docs/GUIDE-vllm.md)**
(credits, pinned commits, our two compatibility patches, tuned profile, systemd unit, opencode config).
Measurements: [`results/2026-11-23-vllm-exl3.md`](results/2026-11-23-vllm-exl3.md). The rest of this README documents the
llama.cpp investigation, which is still useful for profiling and as a portable fallback.

### llama.cpp (fallback)

| | |
|---|---|
| Best backend | **Vulkan** (Unsloth's llama.cpp prebuild); SYCL is 3× slower on prefill even after the fix below |
| Generation | **~20–22 t/s** |
| Prefill | **~610 t/s** (llama-bench, 4K tokens, `-ub 1024`); ~320–400 t/s through the server |
| Max context that fits | **384K tokens** with KV q8_0 (29.1 GiB VRAM); the model supports 1M |

Full numbers: [`results/2026-11-23-arc-pro-b70.md`](results/2026-11-23-arc-pro-b70.md).

## Quick start

```bash
scripts/get-vulkan-build.sh                     # Unsloth llama.cpp Vulkan prebuild -> ./llama-vulkan
LAYOUT=unsloth scripts/download.sh              # UD-Q4_K_XL GGUF -> ./models (sha256-verified)
HOST=0.0.0.0 PORT=8080 scripts/serve.sh         # 384K ctx, KV q8_0, 1 slot
python3 bench/gen_bench.py http://127.0.0.1:8080
```

- systemd unit: [`deploy/systemd/inkling.service`](deploy/systemd/inkling.service)
- opencode provider: [`deploy/opencode/opencode.json`](deploy/opencode/opencode.json)

## What we learned

1. **XMX matters for prefill, not decode.** Vulkan uses the matrix cores (`KHR_coopmat`). Disabling them cuts
   prefill from 588 to 244 t/s, but generation stays the same.
2. **Decode is limited by many small kernels, not by memory bandwidth.** One token is 50 ms of GPU time over
   6,646 dispatches. The largest chunk is the Q8_0 attention mat-vec at ~26 % of peak bandwidth. Unsloth keeps
   attention at Q8_0 in every quant, and the looped layers read it 3× per token.
3. **`-ub 1024` gives +10 % prefill** for free. Larger ubatch does not help, and it costs context (see 5).
4. **DSpark speculative decoding is slower *in llama.cpp*** (19.7 → 12.8–18.9 t/s), while it is a large win
   *in the vLLM recipe* (+~50 % on code). Same drafter, different implementation: in llama.cpp each drafted token
   is verified with the generic MoE path, so every token pulls its own 8 experts and the extra weight reads cost
   more than the accepted tokens save. The vLLM recipe verifies the whole draft block in fused XMX kernels inside
   captured XPU graphs, with an adaptive verifier that stops proposing on low confidence.
5. **Context is limited by the compute buffer, not only by the KV cache.** Relative-attention intermediates scale
   with *context × ubatch* (~23 bytes each), so 640K with `-ub 1024` asks for a 15.5 GB buffer. KV q8_0 at 384K
   with `-ub 512` is the largest config that fits on 32 GB. KV q4_0 should reach ~512K, but that is untested.
6. **SYCL falls back to the CPU on two ops.** `SCALE` on non-contiguous tensors and Inkling's `FLASH_ATTN`
   variant both run on the CPU, which produces 290 GPU↔CPU splits per graph.
   [`patches/0001-ggml-sycl-scale-non-contiguous.patch`](patches/0001-ggml-sycl-scale-non-contiguous.patch) adds a
   strided SCALE kernel, with tests. Splits go from 194 to 2 with `-fa 0` and prefill improves by 42 %, but SYCL
   is still behind Vulkan. The same fix is in review upstream:
   [ggml-org/llama.cpp#32784](https://github.com/ggml-org/llama.cpp/pull/32784).

## Open ideas / help wanted

- Faster Q8_0 / NVFP4 mat-vec shaders for Xe2 in ggml-vulkan. That is where decode time goes.
- Operator fusion for the looped-layer controllers (hundreds of tiny `MUL`/`SCALE`/`CONCAT` per token).
- SYCL support for Inkling's flash-attention variant (relative bias / sinks).
- Multi-user serving: the vLLM recipe is single-sequence today.

## Limitations

- **Quality is measured against the native weights dequantized to BF16** (run on a rented NVIDIA GPU, since the
  native checkpoint does not run on Intel GPUs yet). On wikitext-2: the GGUF Q4 has 97.7 % top-1 agreement and
  KLD 0.0028, close to the cross-engine floor set by UD-Q8_K_XL, which stores the native
  precision (97.9 %, KLD 0.0026); the vLLM EXL3 build is +0.24 % PPL with 96.3 % top-1 agreement
  (this also includes vLLM's FP8 KV cache). Details:
  [`results/2026-11-24-quality-native.md`](results/2026-11-24-quality-native.md); the earlier comparison against
  Q8 is in [`results/2026-11-24-quality.md`](results/2026-11-24-quality.md).
- **Single user only.** No throughput-under-concurrency numbers.
- **One card, one host.** Results may differ with other power caps (ours is 230 W), drivers or kernels.

## Reproduce the measurements

```bash
BENCH=./llama-vulkan/llama-bench MODEL=./models/Inkling-2-Mini-UD-Q4_K_XL.gguf bench/run-bench.sh
GGML_VK_PERF_LOGGER=1 ./llama-vulkan/llama-bench -m MODEL -ngl 99 -fa 1 -p 0 -n 8 -r 1 2> perf.log
python3 bench/vk_profile.py perf.log
```

SYCL: [`docker/Dockerfile.sycl`](docker/Dockerfile.sycl) (upstream + SCALE patch, oneAPI 2026.0). Built end-to-end
and tested on our B70: `docker run --rm --device /dev/dri <image> /opt/llama.cpp/build/bin/test-backend-ops -o SCALE -b SYCL0`
→ 9/9 OK, including the non-contiguous (`v=1`) cases. Compilation is memory-hungry (~3–4 GB RAM per job): use
`--build-arg BUILD_JOBS=2` on a 32 GB host, and do not build it on a host that is close to its memory limit.

## About Hoopytech

<a href="https://hoopytech.com"><b>Hoopytech</b></a> is a boutique AI consultancy. We build AI solutions (LLMs,
agents, RAG, MLOps), embed senior AI specialists and fractional AI leadership in client teams, and help with
EU AI Act compliance. Running capable open models on your own hardware, like this setup, is part of that work.

If you want a private LLM deployment on your infrastructure, or a hand tuning one: **[hoopytech.com](https://hoopytech.com)**.

## License

Scripts and patch: MIT. Model weights: see the model card.
