"""Aggregate GGML_VK_PERF_LOGGER output into GPU time per decoded token, by op.

Usage: GGML_VK_PERF_LOGGER=1 llama-bench -m model.gguf -ngl 99 -fa 1 -p 0 -n 8 -r 1 2> perf.log
       python3 bench/vk_profile.py perf.log
"""
import re
import sys
from collections import defaultdict

blocks = open(sys.argv[1]).read().split("Vulkan Timings:")[1:]
block = blocks[-2] if len(blocks) > 2 else blocks[-1]  # a steady-state decode step
total, count = defaultdict(float), defaultdict(int)
for m in re.finditer(r"^(.+?): (\d+) x ([\d.]+) us = ([\d.]+) us", block, re.M):
    op = re.sub(r" m=\d+.*|\(.*", "", m.group(1)).strip()
    total[op] += float(m.group(4))
    count[op] += int(m.group(2))
T, n = sum(total.values()), sum(count.values())
print(f"GPU time/token = {T / 1000:.1f} ms (~{1e6 / T:.0f} t/s GPU-bound), {n} dispatches, {T / n:.1f} us/dispatch")
for op, t in sorted(total.items(), key=lambda x: -x[1])[:15]:
    print(f"{op[:60]:60} {count[op]:5}x {t / 1000:7.2f} ms {100 * t / T:5.1f}%")
