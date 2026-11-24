# Quality vs. the native weights (2026-11-24)

Follow-up to [`2026-11-24-quality.md`](2026-11-24-quality.md), which used Unsloth's UD-Q8_K_XL as the reference
because the native checkpoint does not run on Intel GPUs. Here the reference is the **native checkpoint**.

**Reference.** `thinkingmachines/Inkling-2-Mini` at revision `7a67ae2`, run once on a rented NVIDIA RTX PRO 6000
with transformers 5.24.0 / torch 2.15.0+cu132. transformers does not implement the ModelOpt NVFP4/MXFP8 kernels;
it **dequantizes the experts to BF16 at load**. So the reference is the native weights in BF16, not the NVFP4
kernel path. Logits were written in llama.cpp's `--kl-divergence-base` format with
[`bench/teacher_logits.py`](../bench/teacher_logits.py), on the **same token ids and positions** as the earlier
run (wikitext-2 test, 150 chunks × 512, second half of each chunk scored, 38,250 positions).
Native PPL: **6.9436**. Sanity check: `llama-perplexity` reads the file back and reports the same 6.9436.

**Candidates** were measured on our Arc Pro B70: GGUFs with `llama-perplexity --kl-divergence` (llama.cpp Vulkan,
Unsloth build b12631, full distributions); the EXL3 build through the vLLM server with `prompt_logprobs`
([`bench/quality_vs_kld_base.py`](../bench/quality_vs_kld_base.py) method).

## Results

| Build | Engine | PPL | PPL vs native | Top-1 agreement with native | Mean KLD vs native |
|---|---|---|---|---|---|
| Native weights (BF16) | transformers, NVIDIA | 6.944 | — | — | — |
| UD-Q8_K_XL | llama.cpp Vulkan | 6.945 | **+0.02 %** | **97.9 %** | **0.0026** (median 0.0010, p99 0.026) |
| UD-Q4_K_XL | llama.cpp Vulkan | 6.940 | **−0.05 %** | **97.7 %** | **0.0028** (median 0.0012, p99 0.026) |
| EXL3 4bpw | vLLM XPU (B70 recipe, FP8 KV) | 6.960 | **+0.24 %** | **96.3 %** | not computed* |

\* The server returns only the top token per position, not the full distribution.

## Reading the numbers

- **The GGUFs are close to the native model**, and Q4 is almost as close as Q8 (KLD 0.0028 vs 0.0026).
  Q8 is not lossless against the native weights: ~2 % of positions pick a different top token, partly because
  the comparison also crosses engines and GPUs (NVIDIA BF16 vs Intel Vulkan).
- A PPL slightly below the reference (Q4) is noise on 38k positions, not an improvement; KLD and top-1 agreement
  are the better measures.
- **The fast vLLM build costs a bit more**: +0.24 % PPL and 96.3 % top-1 agreement. This includes vLLM's
  numerics and its FP8 KV cache, not only the EXL3 weights.
- **Limits:** wikitext perplexity is a proxy, not a task benchmark. Single run per build, one card.

---
Measured by [Hoopytech](https://hoopytech.com).
