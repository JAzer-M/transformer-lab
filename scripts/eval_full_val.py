"""在完整验证集上给一个训练好的存档打分，写出 full_val.json（M3 消融的最终比较用它）。

用法：
    uv run python scripts/eval_full_val.py --ckpt runs/m3/<变体>/seed<种子>/ckpt.pt \
        --out runs/m3/<变体>/seed<种子>/full_val.json
验证集按存档配置里的 data_dir 读取，窗口长度用存档的 block_size。
"""

import argparse
import json
from pathlib import Path

import torch

from tlab.device import get_device
from tlab.eval.full_val import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="写入的 full_val.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    result = evaluate_checkpoint(args.ckpt, args.batch_size, device)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # 手册要求核对 token 数和存档步数，打印出来不用再去翻 json
    print(
        f"完整验证集 val loss {result['val_loss']:.6f}"
        f"（{result['n_tokens']:,} 个 token，第 {result['ckpt_step']} 步的存档）"
    )


if __name__ == "__main__":
    main()
