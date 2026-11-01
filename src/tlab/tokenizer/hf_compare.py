"""把手写 BPE 与 HF tokenizers 的字节级 BPE 放在同一份语料上对照。

两边都从 256 个字节出发、都不做按词预分词（HF 的 ByteLevel 关闭 use_regex），
同样的词表大小下比较两件事：能否无损还原、压缩率（每个 token 平均多少字节）。
并列时的选择规则不同，所以不要求逐个 token 一致。
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tlab.tokenizer.bpe import BPETokenizer


def train_hf_bpe(text: str, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    # 字节级：每个字节映射成一个可见字符；不按空格、标点切词
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def compare_with_hf(train_text: str, eval_text: str, vocab_size: int) -> dict:
    """在 train_text 上训练两种 BPE，在 eval_text 上比较还原与压缩率。"""
    ours = BPETokenizer()
    ours.train(train_text, vocab_size)
    hf = train_hf_bpe(train_text, vocab_size)

    n_bytes = len(eval_text.encode("utf-8"))
    ours_ids = ours.encode(eval_text)
    hf_ids = hf.encode(eval_text).ids

    ours_bpt = n_bytes / len(ours_ids)
    hf_bpt = n_bytes / len(hf_ids)
    return {
        "eval_bytes": n_bytes,
        "ours_tokens": len(ours_ids),
        "hf_tokens": len(hf_ids),
        "ours_bytes_per_token": ours_bpt,
        "hf_bytes_per_token": hf_bpt,
        "relative_diff": abs(ours_bpt - hf_bpt) / hf_bpt,
        "ours_round_trip": ours.decode(ours_ids) == eval_text,
        "hf_round_trip": hf.decode(hf_ids) == eval_text,
        "ours_vocab_size": len(ours.vocab),
        "hf_vocab_size": hf.get_vocab_size(),
    }
