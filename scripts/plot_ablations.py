"""画 M3 消融的 loss 曲线：每组一张 PNG，把各变体叠在一起。

用法：
    uv run python scripts/plot_ablations.py experiments/m3_ablations \
        --configs configs/ablations --out experiments/m3_ablations/figures
"""

import argparse
from pathlib import Path

from tlab.eval.plots import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="各变体结果所在的目录")
    parser.add_argument("--out", type=Path, required=True, help="图片写入的目录")
    parser.add_argument(
        "--configs", type=Path, default=None, help="给了就把结果与这里的配置文件核对"
    )
    args = parser.parse_args()

    for path in plot_all(args.root, args.out, config_dir=args.configs):
        print(path)


if __name__ == "__main__":
    main()
