"""速度与显存短测（A4、A7）：训练吞吐、生成吞吐（有无 KV cache）、缓存大小。

用随机权重和随机 token：这些数字只和结构有关，和参数的具体取值无关，所以不需要存档，
也不读数据。每项测 repeats 次取中位数；显存峰值只在 GPU 上有，CPU、MPS 上是 None。

缓存占多少内存有两个数字：cache_bytes_full_window 是存满一个窗口时 k、v 的字节数，
由形状和类型决定，算一次就够；实际显存看 generate_cache 的 peak_memory_mb。
"""

import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from statistics import median
from typing import Any

import torch
from torch import nn

from tlab.eval.generate import generate
from tlab.model.attention import KVCache
from tlab.model.gpt import GPT, count_parameters
from tlab.train.config import TrainConfig
from tlab.train.utils import (
    autocast_context,
    cross_entropy_loss,
    git_commit,
    gpt_config_from,
    peak_memory_mb,
)


def gpu_temperature_c() -> float | None:
    """当前 GPU 温度；没有 nvidia-smi（比如 Mac）或读取失败时返回 None。"""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return float(done.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def median_of_runs(
    measure: Callable[[], dict[str, float | None]], repeats: int
) -> dict[str, Any]:
    """测 repeats 次，返回吞吐和显存峰值的中位数，每一次的原始结果也留着。"""
    runs = [measure() for _ in range(repeats)]
    peaks = [run["peak_memory_mb"] for run in runs]
    return {
        "tokens_per_sec": median(run["tokens_per_sec"] for run in runs),
        "peak_memory_mb": None if None in peaks else median(peaks),
        "runs": runs,
    }


def _synchronize(device: torch.device) -> None:
    # GPU 的计算是异步的，不等它算完，计时就只算到了提交任务
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(device: torch.device, work: Callable[[], int]) -> dict[str, float | None]:
    """跑一次 work（返回处理了多少 token），得到吞吐、显存峰值和开始时的温度。"""
    temperature = gpu_temperature_c()  # 在计时之外读，不算进耗时
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()
    n_tokens = work()
    _synchronize(device)
    seconds = time.perf_counter() - start
    return {
        "tokens_per_sec": n_tokens / seconds,
        "peak_memory_mb": peak_memory_mb(device),
        "temperature_c": temperature,
    }


def measure_training(
    model: nn.Module,
    config: TrainConfig,
    vocab_size: int,
    steps: int,
    device: torch.device,
) -> dict[str, float | None]:
    """和正式训练相同的一步：累积 grad_accum_steps 个 micro batch，裁剪梯度，再更新。

    只有 token 是随机生成的，所以不含读数据的时间。
    """
    B, T, K = config.batch_size, config.block_size, config.grad_accum_steps
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(K):
            x = torch.randint(0, vocab_size, (B, T), device=device)  # (B, T)
            y = torch.randint(0, vocab_size, (B, T), device=device)  # (B, T)
            with autocast_context(device):
                loss = cross_entropy_loss(model(x), y) / K  # ()
            loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    # 第一步要分配优化器状态，不算进吞吐
    step()

    def work() -> int:
        for _ in range(steps):
            step()
        return steps * K * B * T

    result = _timed(device, work)
    # 设成 None 才真正释放梯度，不然会算进后面生成的显存峰值
    optimizer.zero_grad(set_to_none=True)
    return result


def measure_generation(
    model: nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    use_cache: bool,
    device: torch.device,
) -> dict[str, float | None]:
    """贪心生成 new_tokens 个 token，吞吐只算新生成的。"""

    def work() -> int:
        with autocast_context(device):
            generate(
                model, prompt, new_tokens, top_k=1, use_cache=use_cache
            )  # (B, T + new_tokens)
        return new_tokens * prompt.size(0)

    return _timed(device, work)


def fill_cache(
    model: nn.Module, vocab_size: int, block_size: int, device: torch.device
) -> list[KVCache]:
    """把缓存存满一个窗口：B=1、T=block_size。"""
    idx = torch.randint(0, vocab_size, (1, block_size), device=device)  # (1, T)
    cache = model.new_cache()
    with torch.no_grad(), autocast_context(device):
        model(idx, cache=cache)  # 每层的 k、v 存成 (1, g, T, d)
    return cache


def cache_bytes(cache: list[KVCache]) -> int:
    """所有 k、v 占的字节数。RoPE 的 k 旋转后是 fp32，所以要按实际类型算。"""
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in (layer.k, layer.v)
    )


def bench_variant(
    config: TrainConfig,
    vocab_size: int,
    device: torch.device,
    repeats: int,
    train_steps: int,
    prompt_len: int,
    new_tokens: int,
) -> dict[str, Any]:
    """按配置建一个随机权重的模型，测完所有项。"""
    # 窗口满了以后缓存每一步都要重建，测出来的就不是缓存的速度
    if prompt_len + new_tokens > config.block_size:
        raise ValueError(
            f"prompt_len + new_tokens = {prompt_len + new_tokens} "
            f"不能超过 block_size = {config.block_size}"
        )
    torch.manual_seed(config.seed)
    config_dict = asdict(config)
    model = GPT(gpt_config_from(config_dict, vocab_size)).to(device)
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)  # (1, T)
    temperature_start = gpu_temperature_c()

    train = median_of_runs(
        lambda: measure_training(model, config, vocab_size, train_steps, device),
        repeats,
    )
    # 同样的开头，先测不用缓存，再测用缓存
    no_cache = median_of_runs(
        lambda: measure_generation(model, prompt, new_tokens, False, device), repeats
    )
    cached = median_of_runs(
        lambda: measure_generation(model, prompt, new_tokens, True, device), repeats
    )

    cache = fill_cache(model, vocab_size, config.block_size, device)
    return {
        "config": config_dict,
        "git_commit": git_commit(),
        "n_params": count_parameters(model),
        "device": str(device),
        "repeats": repeats,
        "train_steps": train_steps,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "train": train,
        "generate_no_cache": no_cache,
        "generate_cache": cached,
        "cache_bytes_full_window": cache_bytes(cache),
        "cache_dtype": {"k": str(cache[0].k.dtype), "v": str(cache[0].v.dtype)},
        "temperature_c": {"start": temperature_start, "end": gpu_temperature_c()},
    }


def _cell(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def render_table(results: dict[str, dict[str, Any]]) -> str:
    """把各变体的短测结果排成一张 Markdown 表。"""
    header = [
        "变体",
        "参数量",
        "训练 tokens/s",
        "训练峰值 MiB",
        "生成 tokens/s（无缓存）",
        "生成 tokens/s（有缓存）",
        "提速倍数",
        "生成峰值 MiB（无缓存）",
        "生成峰值 MiB（有缓存）",
        "完整缓存 MiB",
        "缓存类型（k / v）",
        "温度 °C（前/后）",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for name, r in results.items():
        no_cache, cached = r["generate_no_cache"], r["generate_cache"]
        speedup = cached["tokens_per_sec"] / no_cache["tokens_per_sec"]
        temperature = r["temperature_c"]
        row = [
            name,
            f"{r['n_params']:,}",
            _cell(r["train"]["tokens_per_sec"]),
            _cell(r["train"]["peak_memory_mb"]),
            _cell(no_cache["tokens_per_sec"]),
            _cell(cached["tokens_per_sec"]),
            f"{speedup:.2f}",
            _cell(no_cache["peak_memory_mb"]),
            _cell(cached["peak_memory_mb"]),
            f"{r['cache_bytes_full_window'] / 2**20:.2f}",
            f"{r['cache_dtype']['k']} / {r['cache_dtype']['v']}",
            f"{_cell(temperature['start'])} / {_cell(temperature['end'])}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
