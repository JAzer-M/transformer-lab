"""手写 BPE 与 HF tokenizers 对照：同语料、同词表大小，比较无损还原与压缩率。

用法：uv run python scripts/compare_bpe.py
默认在 Tiny Shakespeare 前 100,000 个字符上训练，在最后 50,000 个字符上评估。
验收标准：两边都能无损还原，压缩率相差不超过 5%。
"""

import argparse
import json
import time
from pathlib import Path

from tlab.tokenizer.hf_compare import compare_with_hf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text", type=Path, default=Path("data/shakespeare_char/input.txt")
    )
    parser.add_argument("--train-chars", type=int, default=100_000)
    parser.add_argument("--eval-chars", type=int, default=50_000)
    parser.add_argument("--vocab-size", type=int, default=256 + 500)
    args = parser.parse_args()

    text = args.text.read_text(encoding="utf-8")
    train_text = text[: args.train_chars]
    eval_text = text[-args.eval_chars :]

    start = time.perf_counter()
    result = compare_with_hf(train_text, eval_text, args.vocab_size)
    result["seconds"] = time.perf_counter() - start
    result["passed"] = (
        result["ours_round_trip"]
        and result["hf_round_trip"]
        and result["relative_diff"] <= 0.05
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
