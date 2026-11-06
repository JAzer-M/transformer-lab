"""速度与显存短测：对每个配置测训练吞吐、生成吞吐（有无 KV cache）和缓存大小。

用随机权重，不需要存档和数据。结果写到 --out 的 json，旁边再写一份同名的 .md 表格。

用法：
    uv run python scripts/bench_attention.py \
        --configs configs/ablations/a0_control.yaml configs/ablations/a4_gqa2.yaml \
        --out experiments/m3_ablations/bench.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from tlab.device import get_device
from tlab.eval.bench import bench_variant, render_table
from tlab.train.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--vocab-size", type=int, default=8192, help="TinyStories 是 8192"
    )
    parser.add_argument("--repeats", type=int, default=3, help="每项测几次，取中位数")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=480,
        help="默认 32 + 480 = 512，缓存正好存满 TinyStories 的窗口",
    )
    parser.add_argument(
        "--cooldown", type=float, default=60, help="每个配置开始前空闲降温的秒数"
    )
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    results = {}
    for path in args.configs:
        # 第一个配置也要等：短测紧跟在训练之后，不等的话它测的时候最热
        time.sleep(args.cooldown)
        print(f"测 {path.stem}……", flush=True)
        results[path.stem] = bench_variant(
            load_config(path),
            args.vocab_size,
            device,
            repeats=args.repeats,
            train_steps=args.train_steps,
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
        ) | {"cooldown_seconds": args.cooldown}
        # 每测完一个就写一次，中途断了也不丢前面的结果
        table = render_table(results)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        args.out.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
