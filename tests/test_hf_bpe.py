import re
from pathlib import Path

# 正式训练用的 HF 字节级 BPE 词表；待实现模块在各测试内导入

STORIES = [
    "Once upon a time, there was a little cat named Tom. Tom liked to play.",
    "One day, Lily found a big red ball in the park. She was very happy!",
    'The dog said, "Let\'s go home." They walked home together.',
]
CORPUS = STORIES * 30


def trained_tokenizer(vocab_size: int = 300):
    from tlab.tokenizer.hf_bpe import train_hf_tokenizer

    return train_hf_tokenizer(CORPUS, vocab_size)


def test_vocab_size_counts_bytes_eot_and_merges() -> None:
    tokenizer = trained_tokenizer(300)

    # 256 个字节 + 1 个故事结束标记 + 43 次合并
    assert tokenizer.vocab_size == 300


def test_encode_then_decode_restores_text() -> None:
    tokenizer = trained_tokenizer()
    # 换行、连续空格、制表符、引号、中文、emoji：字节级保证任何文本都能还原
    text = 'Tom said, "Hi!"\n\n  The cat   sat.\t你好 🐱'

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_eot_is_one_token_and_kept_by_decode() -> None:
    from tlab.tokenizer.hf_bpe import EOT

    tokenizer = trained_tokenizer()
    text = f"The end.{EOT}Once upon a time"
    ids = tokenizer.encode(text)

    # 结束标记整体是一个 token，不会被拆成 "<"、"|" 等字节
    assert tokenizer.encode(EOT) == [tokenizer.eot_id]
    assert ids.count(tokenizer.eot_id) == 1
    # 解码时保留结束标记，生成样例里能看到故事在哪里结束
    assert tokenizer.decode(ids) == text


def test_merges_stay_inside_words() -> None:
    tokenizer = trained_tokenizer()
    pieces = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]

    # 学到了“空格 + 字母”这样的片段，例如 " t"
    assert any(p.startswith(" ") and len(p) > 1 for p in pieces)
    # GPT-2 正则先按词切开：空格只出现在片段开头，不会跟在字母或标点后面（如 "e "）
    assert not any(re.search(r"\S ", p) for p in pieces)


def test_encode_batch_matches_encode() -> None:
    tokenizer = trained_tokenizer()
    texts = ["Tom ran.", "Lily smiled at the dog.", ""]

    assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]


def test_training_is_reproducible() -> None:
    sample = "Tom and Lily played with the red ball in the park."

    first = trained_tokenizer()
    second = trained_tokenizer()

    assert first.encode(sample) == second.encode(sample)


def test_save_and_load_keep_encoding(tmp_path: Path) -> None:
    from tlab.tokenizer.hf_bpe import HFTokenizer

    tokenizer = trained_tokenizer()
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = HFTokenizer.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.eot_id == tokenizer.eot_id
    sample = "Lily and Tom played with the ball."
    assert loaded.encode(sample) == tokenizer.encode(sample)
