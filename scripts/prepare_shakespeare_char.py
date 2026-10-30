"""下载 Tiny Shakespeare，转成字符编号，存为训练集 / 验证集。

用法：uv run python scripts/prepare_shakespeare_char.py
输出：data/shakespeare_char/{input.txt, train.bin, val.bin, meta.json}
"""

import argparse
import urllib.request
from pathlib import Path

from tlab.data.char import CharTokenizer, save_ids, split_ids

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/shakespeare_char"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只在本地没有原文时下载一次
    raw_path = out_dir / "input.txt"
    if not raw_path.exists():
        print(f"下载 {URL}")
        urllib.request.urlretrieve(URL, raw_path)
    text = raw_path.read_text(encoding="utf-8")

    # 建字表、编码、按顺序切分
    tokenizer = CharTokenizer.from_text(text)
    train, val = split_ids(tokenizer.encode(text), args.val_fraction)

    save_ids(train, out_dir / "train.bin")
    save_ids(val, out_dir / "val.bin")
    tokenizer.save(out_dir / "meta.json")

    print(f"字符总数 {len(text):,}，字表大小 {tokenizer.vocab_size}")
    print(f"训练集 {len(train):,} 个 token，验证集 {len(val):,} 个 token")
    print(f"已写入 {out_dir}")


if __name__ == "__main__":
    main()
