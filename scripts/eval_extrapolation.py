"""在 2 倍训练长度上给存档打分，写出 extrapolation.json（A1 的外推评测）。

只支持正弦和 RoPE；可学习位置表没有超出训练长度的位置，会直接报错。

用法：
    uv run python scripts/eval_extrapolation.py --ckpt runs/m3/<变体>/seed<种子>/ckpt.pt \
        --out runs/m3/<变体>/seed<种子>/extrapolation.json
"""

import argparse
import json
from pathlib import Path

import torch

from tlab.device import get_device
from tlab.eval.full_val import extrapolate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="写入的 extrapolation.json"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    result = extrapolate_checkpoint(args.ckpt, args.batch_size, device)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # 手册要求核对长度和存档步数，打印出来不用再去翻 json
    print(
        f"长度 {result['length']}：前半段 {result['first_half_loss']:.6f}，"
        f"后半段 {result['second_half_loss']:.6f}"
        f"（每半段 {result['n_tokens_per_half']:,} 个 token，第 {result['ckpt_step']} 步的存档）"
    )


if __name__ == "__main__":
    main()
