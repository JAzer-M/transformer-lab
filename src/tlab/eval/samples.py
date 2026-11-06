"""给每个变体的存档生成文字样例（M3 第 6 步），写成一个 Markdown 文件。

和 sample.py 一样不用 autocast；用 KV 缓存生成，文字和不用缓存时相同。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tlab.data.loader import load_tokenizer
from tlab.eval.ablation import CONTROL, GROUPS, REFERENCE, STRESS
from tlab.eval.generate import generate
from tlab.train.utils import load_model

# 全部 12 个变体：对照、单变量、高学习率加测、组合
VARIANTS = [CONTROL, *(n for g in GROUPS.values() for n in g), *STRESS, *REFERENCE]


@dataclass
class SampleParams:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    seeds: list[int]


def sample_checkpoint(
    ckpt_path: Path, params: SampleParams, device: torch.device
) -> dict[str, Any]:
    """按存档重建模型，每个种子生成一段文字；返回存档步数和 {种子: 文字}。"""
    model, ckpt = load_model(ckpt_path, device)
    config = ckpt["config"]
    if ckpt["step"] != config["max_iters"]:
        raise ValueError(
            f"{ckpt_path} 存的是第 {ckpt['step']} 步，不等于 max_iters={config['max_iters']}"
        )
    tokenizer = load_tokenizer(Path(config["data_dir"]))
    prompt = torch.tensor([tokenizer.encode(params.prompt)], device=device)  # (1, T)
    texts = {}
    for seed in params.seeds:
        out = generate(
            model,
            prompt,
            params.max_new_tokens,
            temperature=params.temperature,
            top_k=params.top_k,
            generator=torch.Generator(device=device).manual_seed(seed),
            use_cache=True,
        )  # (1, T + max_new_tokens)
        texts[seed] = tokenizer.decode(out[0].tolist())  # out[0]：(T + max_new_tokens,)
    return {"ckpt_step": ckpt["step"], "texts": texts}


def render_samples(
    results: dict[str, dict[str, Any]],
    params: SampleParams,
    commit: str,
    device: str,
    run_seed: int,
) -> str:
    """排成 Markdown；开头写清复现需要的所有信息。"""
    seeds = "、".join(str(seed) for seed in params.seeds)
    lines = [
        "# M3 生成样例",
        "",
        f"由 `scripts/sample_ablations.py` 生成。commit `{commit}`，设备 `{device}`，"
        f"用每个变体种子 {run_seed} 的最终存档。",
        "",
        f"开头 `{params.prompt}`，新生成 {params.max_new_tokens} 个 token，"
        f"temperature {params.temperature}，top-k {params.top_k}，生成种子 {seeds}，"
        "使用 KV 缓存。",
    ]
    for name, result in results.items():
        lines += ["", f"## {name}（第 {result['ckpt_step']} 步的存档）"]
        for seed, text in result["texts"].items():
            lines += ["", f"生成种子 {seed}：", "", "```text", text, "```"]
    return "\n".join(lines) + "\n"
