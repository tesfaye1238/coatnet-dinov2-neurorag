"""Efficiency measurements on the ACTUAL hardware they are run on.

* Parameters: counted from the instantiated model.
* FLOPs: ``torch.utils.flop_counter.FlopCounterMode`` (counts matmul/conv/attention FLOPs, 1 MAC = 2 FLOPs;
  elementwise ops, norms and activations are not counted). MACs are reported as FLOPs/2. This convention differs
  from some libraries, so values are not directly comparable with other papers' tables.
* Latency: batch-1 forward after warm-up, CUDA synchronised; median and mean over ``iters``.
* Energy is not measured here (no documented measurement method) and is therefore never reported.
* Desktop numbers say nothing about edge-device deployability.
"""
from __future__ import annotations

import statistics
import time

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from neurofusion.utils.env import hardware_info


def count_flops(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 224, 224)) -> dict:
    model.eval()
    x = torch.zeros(*input_size, device=next(model.parameters()).device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    flops = fc.get_total_flops()
    return {"flops": flops, "gflops": flops / 1e9, "gmacs": flops / 2e9, "convention": "FlopCounterMode; 1 MAC = 2 FLOPs; elementwise ops excluded"}


def benchmark_latency(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 224, 224), warmup: int = 10, iters: int = 50, device: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(dev)
    x = torch.randn(*input_size, device=dev)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    out = {
        "device": str(dev), "batch": input_size[0], "warmup": warmup, "iters": iters,
        "latency_ms_median": statistics.median(times), "latency_ms_mean": statistics.fmean(times),
        "latency_ms_std": statistics.pstdev(times),
        "throughput_img_per_s": input_size[0] * 1000.0 / statistics.median(times),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "hardware": hardware_info(),
        "energy": "not measured",
    }
    return out


def efficiency_report(model: nn.Module, **kw) -> dict:
    from neurofusion.models.classifier import count_parameters

    return {"parameters": count_parameters(model), "flops": count_flops(model), "latency": benchmark_latency(model, **kw)}
