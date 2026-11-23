# Inkling 2 Mini on Intel Arc Pro B70

Running [Inkling 2 Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (26B MoE, 5.4B effective active,
looped layers, NVFP4 experts) on a single **Intel Arc Pro B70 (32 GB)** with llama.cpp. The repo has
reproducible scripts, measurements, a profile of where decode time goes, and a SYCL fix.

## TL;DR

**Use vLLM, not llama.cpp, on this card.** The community B70 recipe (vLLM 0.33 XPU, EXL3 4-bit experts, native
SYCL kernels, DSpark) runs **3–6× faster** than the best llama.cpp configuration we found:

| | llama.cpp Vulkan | vLLM XPU + EXL3 + DSpark |
|---|---|---|
| Generation | ~20 t/s | **60 t/s (prose) – 118 t/s (code)** |
| Prefill | ~600 t/s | **~1,400–1,700 t/s** |
| Context | 384K (q8 KV) | 262K (FP8 KV) |

How to reproduce on a fresh host, plus the two compatibility fixes it needed:
[`results/2026-11-23-vllm-exl3.md`](results/2026-11-23-vllm-exl3.md). The rest of this README documents the
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
4. **DSpark speculative decoding is slower here.** Each drafted token routes to its own experts, so verifying
   several tokens multiplies MoE weight reads. Acceptance is also low at the recommended temperature of 1.0.
5. **Context is limited by the compute buffer, not only by the KV cache.** Relative-attention intermediates scale
   with *context × ubatch* (~23 bytes each), so 640K with `-ub 1024` asks for a 15.5 GB buffer. KV q8_0 at 384K
   with `-ub 512` is the largest config that fits on 32 GB. KV q4_0 should reach ~512K, but that is untested.
6. **SYCL falls back to the CPU on two ops.** `SCALE` on non-contiguous tensors and Inkling's `FLASH_ATTN`
   variant both run on the CPU, which produces 290 GPU↔CPU splits per graph.
   [`patches/0001-ggml-sycl-scale-non-contiguous.patch`](patches/0001-ggml-sycl-scale-non-contiguous.patch) adds a
   strided SCALE kernel, with tests. Splits go from 194 to 2 with `-fa 0` and prefill improves by 42 %, but SYCL
   is still behind Vulkan.

## Open ideas / help wanted

- Faster Q8_0 / NVFP4 mat-vec shaders for Xe2 in ggml-vulkan. That is where decode time goes.
- Operator fusion for the looped-layer controllers (hundreds of tiny `MUL`/`SCALE`/`CONCAT` per token).
- SYCL support for Inkling's flash-attention variant (relative bias / sinks).
- Upstreaming the SYCL SCALE patch to ggml-org/llama.cpp.

## Reproduce the measurements

```bash
BENCH=./llama-vulkan/llama-bench MODEL=./models/Inkling-2-Mini-UD-Q4_K_XL.gguf bench/run-bench.sh
GGML_VK_PERF_LOGGER=1 ./llama-vulkan/llama-bench -m MODEL -ngl 99 -fa 1 -p 0 -n 8 -r 1 2> perf.log
python3 bench/vk_profile.py perf.log
```

SYCL: [`docker/Dockerfile.sycl`](docker/Dockerfile.sycl) (upstream + patch, oneAPI 2026.0). Our measurements used an
equivalent local oneAPI image, and this public Dockerfile has not been built end-to-end yet.

## License

Scripts and patch: MIT. Model weights: see the model card.
