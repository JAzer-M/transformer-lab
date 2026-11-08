"""F1：在 GSM8K 固定的题目上评测模型，写出每题的记录和整体准确率。

用法（第一次会下载基座权重，约 1.2 GB，放在 HuggingFace 默认缓存目录）：
    uv run python scripts/prepare_gsm8k.py
    uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_fewshot.yaml \
        --out runs/m5/f1_base_fewshot.json
冒烟时在命令后面加覆盖项，比如 limit=5 device=cpu。

环境里设了 SOCKS 代理（ALL_PROXY=socks5://...）时，下载会报缺 socksio。如果同时有 HTTP
代理，在命令前加 env -u ALL_PROXY -u all_proxy，这条命令就只走 HTTP 代理。
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from tlab.convert.qwen3 import MODEL_FILES, load_qwen3
from tlab.device import get_device
from tlab.eval.gsm8k import (
    Example,
    evaluate_gsm8k,
    few_shot_prompt,
    load_eval_config,
    load_examples,
    zero_shot_prompt,
)
from tlab.model.gpt import count_parameters
from tlab.model.layers import RMSNorm
from tlab.tokenizer.hf_bpe import HFTokenizer
from tlab.train.utils import git_commit


def prompt_builder(mode: str, shots: list[Example]) -> Callable[[str], str]:
    """配置已经检查过 mode，这里只有两种。"""
    if mode == "few_shot":
        return lambda question: few_shot_prompt(question, shots)
    return zero_shot_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None, help="不给就下载")
    parser.add_argument("overrides", nargs="*", help="覆盖单个字段，如 limit=5")
    args = parser.parse_args()

    config = load_eval_config(args.config, args.overrides)
    data_dir = Path(config.data_dir)
    subset = json.loads(Path(config.subset_file).read_text(encoding="utf-8"))
    test = load_examples(data_dir / "test.jsonl")
    train = load_examples(data_dir / "train.jsonl")
    questions = [(i, test[i]) for i in subset["test_indices"][: config.limit]]
    shots = [train[i] for i in subset["few_shot_indices"]]

    # 先建好目录：不建的话要等评测完才报错
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 下载到的目录名就是这份权重在 HF 上的版本号
    model_dir = args.model_dir or Path(
        snapshot_download(config.repo, allow_patterns=MODEL_FILES)
    )
    model_name = (
        str(args.model_dir) if args.model_dir else f"{config.repo}@{model_dir.name}"
    )
    device = get_device() if config.device == "auto" else torch.device(config.device)
    model = load_qwen3(model_dir, device)
    tokenizer = HFTokenizer.load(model_dir / "tokenizer.json")

    result = evaluate_gsm8k(
        model,
        tokenizer,
        questions,
        prompt_builder(config.prompt_mode, shots),
        config.max_new_tokens,
        config.eos_token_id,
        device,
        log_every=50,
    )
    report = {
        "config": asdict(config),
        "subset_seed": subset["seed"],
        "git_commit": git_commit(),
        "model": model_name,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        # bf16 下会影响结果；第 1 批的 F1 是在加这个开关之前跑的，那时是关闭的
        # 开关对所有 RMSNorm 一起设置；用 any：模型里没有 RMSNorm 时是 False
        "rmsnorm_fp32": any(
            m.compute_fp32 for m in model.modules() if isinstance(m, RMSNorm)
        ),
        "n_params": count_parameters(model),
        "torch_version": torch.__version__,
        **result,
    }
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{config.prompt_mode} 准确率 {result['accuracy']:.4f}"
        f"（{result['correct']}/{result['total']}），"
        f"{result['tokens_per_second']:.1f} tokens/s，用时 {result['seconds']:.0f} 秒"
    )


if __name__ == "__main__":
    main()
