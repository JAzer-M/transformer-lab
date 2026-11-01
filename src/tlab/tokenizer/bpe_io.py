"""手写 BPE 的保存与加载：只存合并表，词表按合并顺序重建。"""

import json
from pathlib import Path

from tlab.tokenizer.bpe import BPETokenizer


def save_bpe(tokenizer: BPETokenizer, path: Path) -> None:
    # 按学到的先后顺序存成 [a, b, 新编号] 列表；JSON 不支持元组作为键
    merges = [[a, b, idx] for (a, b), idx in tokenizer.merges.items()]
    path.write_text(json.dumps({"merges": merges}), encoding="utf-8")


def load_bpe(path: Path) -> BPETokenizer:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = BPETokenizer()
    # 按原顺序重放合并规则：新 token 的字节 = 两个旧 token 的字节拼接
    for a, b, idx in data["merges"]:
        tokenizer.merges[(a, b)] = idx
        tokenizer.vocab[idx] = tokenizer.vocab[a] + tokenizer.vocab[b]
    return tokenizer
