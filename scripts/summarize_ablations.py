"""把 M3 各次消融训练的结果汇总成一张 Markdown 表。

用法：
    uv run python scripts/summarize_ablations.py experiments/m3_ablations \
        --configs configs/ablations --out experiments/m3_ablations/SUMMARY.md
目录结构：<root>/<变体>/seed<种子>/{run.json, metrics.csv}
<root> 下有 bench.json、bench_batch/ 时，表的最后再加上短测的结果。
"""

import argparse
from pathlib import Path

from tlab.eval.ablation import (
    load_bench,
    load_cache_speed,
    load_variants,
    render_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="各变体结果所在的目录")
    parser.add_argument("--out", type=Path, default=None, help="写入的 SUMMARY.md 路径")
    parser.add_argument(
        "--configs", type=Path, default=None, help="给了就把结果与这里的配置文件核对"
    )
    args = parser.parse_args()

    text = render_summary(
        load_variants(args.root, config_dir=args.configs),
        load_bench(args.root),
        load_cache_speed(args.root),
    )
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
