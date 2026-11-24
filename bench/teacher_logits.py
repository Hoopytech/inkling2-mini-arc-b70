"""Reference logits from the native checkpoint, written in the llama.cpp KL-divergence base format.

Needs only torch + transformers + numpy and a GPU that fits the BF16-dequantized model:

    python teacher_logits.py --model DIR --tokens TOKENS.bin --out OUT_DIR [--device cuda]

``TOKENS.bin`` is the header + token ids of an existing base file (n_ctx, n_vocab, n_chunk as int32, then
n_chunk*n_ctx int32 tokens), so the teacher scores exactly the same tokens and positions as the quants.

For every chunk, positions ``n_ctx//2 .. n_ctx-2`` are scored (as llama-perplexity does). Log-probabilities over
the first ``n_vocab`` ids are quantized to uint16 exactly like llama.cpp's ``log_softmax`` (perplexity.cpp):
``scale = (max - min) / 65535``, ``min_log_prob = min - max - log(sum(exp(l - max)))``, value
``round((l - min) / scale)`` for ``l > min`` else 0. Each chunk is saved as its own file, so a preempted run
resumes; ``--assemble`` concatenates them into a single base file usable by ``llama-perplexity
--kl-divergence`` and by ``bench/quality_vs_kld_base.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np


def read_tokens(path: Path) -> tuple[int, int, int, np.ndarray]:
    data = path.read_bytes()
    n_ctx, n_vocab, n_chunk = struct.unpack("<iii", data[:12])
    tokens = np.frombuffer(data[12:], dtype=np.int32)
    assert tokens.size == n_ctx * n_chunk, "token count does not match the header"
    return n_ctx, n_vocab, n_chunk, tokens


def quantize_rows(logits, n_vocab: int):  # type: ignore[no-untyped-def]
    """llama.cpp-compatible uint16 log-prob rows. ``logits``: float32 tensor [rows, >= n_vocab] on device."""
    import torch

    lg = logits[:, :n_vocab].float()
    mx = lg.max(dim=1, keepdim=True).values
    mn = lg.min(dim=1, keepdim=True).values
    lse = torch.log(torch.exp(lg - mx).sum(dim=1, keepdim=True))
    scale = (mx - mn) / 65535.0
    min_lp = mn - mx - lse
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.where(lg > mn, torch.round(inv * (lg - mn)), torch.zeros_like(lg)).clamp(0, 65535)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    out = np.zeros((lg.shape[0], nv), dtype=np.uint16)
    head = torch.cat([scale, min_lp], dim=1).cpu().numpy().astype(np.float32)
    out[:, :4] = head.view(np.uint16)
    out[:, 4 : 4 + n_vocab] = q.to(torch.int32).cpu().numpy().astype(np.uint16)
    return out


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM

    n_ctx, n_vocab, n_chunk, tokens = read_tokens(Path(args.tokens))
    out = Path(args.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    first = n_ctx // 2
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(args.model, device_map=args.device, dtype=torch.bfloat16)
    model.eval()
    load_s = time.time() - t0
    qcfg = getattr(model.config, "quantization_config", None)
    meta = {
        "model_dir": args.model,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "quantization_config": qcfg.to_dict() if qcfg is not None and hasattr(qcfg, "to_dict") else qcfg,
        "load_seconds": round(load_s, 1),
        "n_ctx": n_ctx,
        "n_vocab": n_vocab,
        "n_chunk": n_chunk,
        "linear_types": sorted(
            {type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__ or "Expert" in type(m).__name__}
        ),
    }
    print(json.dumps(meta, indent=1, default=str), flush=True)
    nll = 0.0
    count = 0
    for c in range(n_chunk):
        dest = out / "chunks" / f"{c:05d}.npy"
        ids = torch.tensor(tokens[c * n_ctx : (c + 1) * n_ctx], dtype=torch.long, device=args.device)[None]
        if dest.exists() and not args.score_only:
            continue
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]  # [n_ctx, vocab_padded]
        rows = quantize_rows(logits[first : n_ctx - 1], n_vocab)
        np.save(dest, rows)
        lp = torch.log_softmax(logits[first : n_ctx - 1, :n_vocab].float(), dim=1)
        tgt = ids[0, first + 1 : n_ctx]
        nll -= lp.gather(1, tgt[:, None]).sum().item()
        count += tgt.numel()
        if (c + 1) % 10 == 0:
            print(
                f"chunk {c + 1}/{n_chunk}  ppl so far {math.exp(nll / count):.4f}  elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
    if count:
        meta["teacher_ppl_this_session"] = math.exp(nll / count)
        meta["positions_this_session"] = count
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("done", json.dumps({k: meta[k] for k in meta if "ppl" in k or "positions" in k}))


def assemble(args: argparse.Namespace) -> None:
    n_ctx, n_vocab, n_chunk, tokens = read_tokens(Path(args.tokens))
    out = Path(args.out)
    with open(out / "teacher-base.kld", "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        f.write(tokens.tobytes())
        for c in range(n_chunk):
            f.write(np.load(out / "chunks" / f"{c:05d}.npy").tobytes())
    print("assembled", (out / "teacher-base.kld").stat().st_size, "bytes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=False)
    p.add_argument("--tokens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--score-only", action="store_true")
    a = p.parse_args()
    assemble(a) if a.assemble else run(a)


if __name__ == "__main__":
    main()
