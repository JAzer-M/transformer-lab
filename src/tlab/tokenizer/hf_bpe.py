"""正式训练用的词表：HF tokenizers 的字节级 BPE（为了速度；原理见手写的 bpe.py）。

与手写版的两点不同：
1. 先用 GPT-2 的正则把文本切成词、数字、标点和空白，合并只在切出的片段内部进行，
   所以学不到 "e " 这种跨词片段；
2. 另加一个特殊 token <|endoftext|>，标记一篇故事结束。
"""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


class HFTokenizer:
    """包一层 tokenizers.Tokenizer，接口与 CharTokenizer 一致：vocab_size、encode、decode。"""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        eot_id = tokenizer.token_to_id(EOT)
        assert eot_id is not None, f"词表里没有 {EOT}"
        self.eot_id: int = eot_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        # 在 Rust 端多线程并行编码，比逐条调用 encode 快得多
        return [encoding.ids for encoding in self._tokenizer.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        # 保留 <|endoftext|>：生成样例里能看到故事在哪里结束
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: Path) -> None:
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "HFTokenizer":
        return cls(Tokenizer.from_file(str(path)))


def train_hf_tokenizer(
    texts: Iterable[str], vocab_size: int, show_progress: bool = False
) -> HFTokenizer:
    """在 texts 上训练字节级 BPE；vocab_size 含 256 个字节和 <|endoftext|>。"""
    tokenizer = Tokenizer(models.BPE())
    # 每个字节映射成一个可见字符；use_regex=True 即先按 GPT-2 的正则切词
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        # 256 个字节全部放进初始词表，没见过的字节也能编码
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return HFTokenizer(tokenizer)
