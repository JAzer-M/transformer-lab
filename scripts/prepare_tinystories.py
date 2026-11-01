"""下载 TinyStories V2，训练 BPE 词表，把故事编码成 uint16 的 train.bin / val.bin。

用法：
    uv run python scripts/prepare_tinystories.py            # 全量，只在 Windows 上跑
    uv run python scripts/prepare_tinystories.py --sample   # 只用 valid 文件，Mac 冒烟
输出：<out-dir>/{tokenizer.json, train.bin, val.bin, info.json}，原文在 <out-dir>/raw/

全量：词表在完整训练集上训练，train.bin 来自训练集，val.bin 来自官方 valid 文件。
--sample：只下载 valid（约 21.5 MiB），前 90% 的故事训练词表并写成 train.bin，
后 10% 写成 val.bin。
"""

import argparse
import json
import time
from pathlib import Path

from tlab.data.tinystories import (
    REVISION,
    SOURCE_FILES,
    count_token,
    download,
    iter_stories,
    write_tokens,
)
from tlab.tokenizer.hf_bpe import train_hf_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="只用 valid 文件")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    default_dir = "data/tinystories_sample" if args.sample else "data/tinystories"
    out_dir: Path = args.out_dir or Path(default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # 下载并核对大小与 SHA-256；已下载过则只核对
    valid_path = download("valid", out_dir / "raw")
    if args.sample:
        stories = list(iter_stories(valid_path))
        n_train = int(len(stories) * 0.9)
        tokenizer_texts = stories[:n_train]
        train_stories, val_stories = stories[:n_train], stories[n_train:]
        sources = ["valid"]
    else:
        train_path = download("train", out_dir / "raw")
        # 训练集 2 GB：词表训练和编码各从头读一遍文件，不整个放进内存
        tokenizer_texts = iter_stories(train_path)
        train_stories, val_stories = iter_stories(train_path), iter_stories(valid_path)
        sources = ["train", "valid"]
    print(f"下载与核对完成，{time.perf_counter() - start:.1f}s")

    t0 = time.perf_counter()
    tokenizer = train_hf_tokenizer(tokenizer_texts, args.vocab_size, show_progress=True)
    tokenizer.save(out_dir / "tokenizer.json")
    tokenizer_seconds = time.perf_counter() - t0
    print(
        f"词表 {tokenizer.vocab_size}，<|endoftext|> = {tokenizer.eot_id}，"
        f"{tokenizer_seconds:.1f}s"
    )

    t0 = time.perf_counter()
    n_train_tokens = write_tokens(train_stories, tokenizer, out_dir / "train.bin")
    n_val_tokens = write_tokens(val_stories, tokenizer, out_dir / "val.bin")
    encode_seconds = time.perf_counter() - t0

    # 每篇故事末尾恰好一个结束标记，数一数就是故事篇数，顺便检查文件能读回；
    # 分块计数，全量约 5 亿 token 时也不会多占 0.5 GB 内存
    n_train_stories = count_token(out_dir / "train.bin", tokenizer.eot_id)
    n_val_stories = count_token(out_dir / "val.bin", tokenizer.eot_id)

    info = {
        "mode": "sample" if args.sample else "full",
        "revision": REVISION,
        "sources": {s: vars(SOURCE_FILES[s]) for s in sources},
        "vocab_size": tokenizer.vocab_size,
        "eot_id": tokenizer.eot_id,
        "train_tokens": n_train_tokens,
        "val_tokens": n_val_tokens,
        "train_stories": n_train_stories,
        "val_stories": n_val_stories,
        "tokenizer_seconds": tokenizer_seconds,
        "encode_seconds": encode_seconds,
        "total_seconds": time.perf_counter() - start,
    }
    (out_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"训练集 {n_train_stories:,} 篇、{n_train_tokens:,} 个 token；"
        f"验证集 {n_val_stories:,} 篇、{n_val_tokens:,} 个 token"
    )
    print(f"编码 {encode_seconds:.1f}s，已写入 {out_dir}")


if __name__ == "__main__":
    main()
