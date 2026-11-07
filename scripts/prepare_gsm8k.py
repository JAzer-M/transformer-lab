"""下载 GSM8K，把 parquet 转成 jsonl，每行 {"question": ..., "answer": ...}。

pyarrow 只在这个脚本里用，所以放在 dev 依赖；src/tlab 只读转好的 jsonl。

用法（共下载约 2.7 MB）：
    uv run python scripts/prepare_gsm8k.py
输出：<out-dir>/{train.jsonl, test.jsonl}，原文件在 <out-dir>/raw/

环境里设了 SOCKS 代理（ALL_PROXY=socks5://...）时，下载会报缺 socksio。如果同时有 HTTP
代理，在命令前加 env -u ALL_PROXY -u all_proxy，这条命令就只走 HTTP 代理。
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from tlab.data.tinystories import SourceFile, verify_file
from tlab.eval.gsm8k import TEST_TOTAL, TRAIN_TOTAL

REPO_ID = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
SOURCE_FILES = {
    "train": SourceFile(
        "main/train-00000-of-00001.parquet",
        2_306_545,
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    ),
    "test": SourceFile(
        "main/test-00000-of-00001.parquet",
        419_088,
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    ),
}
EXPECTED_ROWS = {"train": TRAIN_TOTAL, "test": TEST_TOTAL}
COLUMNS = ("question", "answer")


def parquet_to_jsonl(parquet: Path, out: Path) -> int:
    """只留题目和答案，返回写出的条数。"""
    # 没有这两列时 select 会报 KeyError
    rows = pq.read_table(parquet).select(list(COLUMNS)).to_pylist()
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def prepare_split(split: str, out_dir: Path) -> Path:
    """准备 train 或 test。原文件已在时不联网，只核对大小和 SHA-256。"""
    source = SOURCE_FILES[split]
    raw_dir = out_dir / "raw"
    path = raw_dir / source.filename
    if not path.exists():
        hf_hub_download(
            REPO_ID,
            source.filename,
            repo_type="dataset",
            revision=REVISION,
            local_dir=raw_dir,
        )
    verify_file(path, source.size, source.sha256)

    out = out_dir / f"{split}.jsonl"
    count = parquet_to_jsonl(path, out)
    if count != EXPECTED_ROWS[split]:
        raise ValueError(f"{split} 应有 {EXPECTED_ROWS[split]} 条，实际 {count} 条")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/gsm8k"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SOURCE_FILES:
        out = prepare_split(split, args.out_dir)
        print(f"{split}：{EXPECTED_ROWS[split]} 条，写到 {out}")


if __name__ == "__main__":
    main()
