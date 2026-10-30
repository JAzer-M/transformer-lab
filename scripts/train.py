"""训练入口。

用法：
    uv run python scripts/train.py --config configs/shakespeare_char.yaml
    uv run python scripts/train.py --config configs/tiny_smoke.yaml max_iters=20
"""

import argparse

from tlab.train.config import load_config
from tlab.train.run import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("overrides", nargs="*", help="覆盖单个字段，如 max_iters=20")
    args = parser.parse_args()

    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
