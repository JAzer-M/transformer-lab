"""从 experiments/m5_gsm8k/ 的原始结果生成 M5 的汇总表。

用法：
    uv run python scripts/summarize_m5.py --out experiments/m5_gsm8k/SUMMARY.md
"""

import argparse
from pathlib import Path

from tlab.eval.m5_summary import render_m5_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/m5_gsm8k"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(render_m5_summary(args.results), encoding="utf-8")
    print(f"写到 {args.out}")


if __name__ == "__main__":
    main()
