import pytest

# 待实现模块在各测试内导入：先写好 get_stats / merge 时，它们的测试可以先单独通过


# ---------- 两个基础函数 ----------


def test_get_stats_counts_adjacent_pairs() -> None:
    from tlab.tokenizer.bpe import get_stats

    # 相邻对：(1,2) (2,3) (3,1) (1,2)
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_get_stats_keeps_first_occurrence_order() -> None:
    from tlab.tokenizer.bpe import get_stats

    # 字典按“第一次出现”的顺序排列；并列时 max 会取最先出现的那一对
    assert list(get_stats([5, 6, 7, 5, 6, 7])) == [(5, 6), (6, 7), (7, 5)]


def test_merge_replaces_pair_left_to_right() -> None:
    from tlab.tokenizer.bpe import merge

    assert merge([1, 2, 3, 1, 2], (1, 2), 256) == [256, 3, 256]
    # 重叠时从左往右、不重复使用：1 1 1 只能合并前两个
    assert merge([1, 1, 1], (1, 1), 256) == [256, 1]
    # 没有这一对时原样返回
    assert merge([4, 5], (1, 2), 256) == [4, 5]


# ---------- 训练 ----------


def test_train_on_textbook_example() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer()
    tokenizer.train("aaabdaaabac", vocab_size=256 + 3)

    # a=97, b=98
    # 第 1 次：(a, a) 出现 4 次 → 256 = "aa"
    # 第 2 次：(256, a) 与 (a, b) 都出现 2 次，取先出现的 (256, a) → 257 = "aaa"
    # 第 3 次：(257, b) 出现 2 次 → 258 = "aaab"
    assert tokenizer.merges == {(97, 97): 256, (256, 97): 257, (257, 98): 258}
    assert tokenizer.vocab[256] == b"aa"
    assert tokenizer.vocab[257] == b"aaa"
    assert tokenizer.vocab[258] == b"aaab"


def test_vocab_size_controls_number_of_merges() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    text = "the cat sat on the mat. the cat ate the rat.\n" * 5
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=256 + 20)

    assert len(tokenizer.merges) == 20
    assert len(tokenizer.vocab) == 256 + 20


def test_train_rejects_vocab_smaller_than_bytes() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    # 256 个字节是基础词表，不能更小
    with pytest.raises(AssertionError):
        BPETokenizer().train("abc", vocab_size=200)


# ---------- 编码与解码 ----------


def test_encode_uses_learned_merges() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer()
    tokenizer.train("aaabdaaabac", vocab_size=256 + 3)

    # d=100, c=99
    assert tokenizer.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    # 新文本也按学到的合并规则编码
    assert tokenizer.encode("aaab") == [258]
    assert tokenizer.encode("") == []


def test_encode_applies_merges_in_learned_order() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    # 手动设定两条合并规则：先学到 (b, c) → 256，后学到 (a, b) → 257
    tokenizer = BPETokenizer()
    tokenizer.merges = {(98, 99): 256, (97, 98): 257}
    tokenizer.vocab[256] = b"bc"
    tokenizer.vocab[257] = b"ab"

    # "abc" 里两对都能合并；必须按学到的先后顺序，先合并 (b, c)
    # 若按出现次数或从左往右挑，会先合并 (a, b) 得到 [257, 99]
    assert tokenizer.encode("abc") == [97, 256]


def test_round_trip_any_text() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer()
    tokenizer.train("Once upon a time, there was a little cat.\n" * 10, 256 + 30)

    # 包括训练时没见过的字符：中文、emoji、换行
    for text in ["Once upon a time.", "你好，世界", "🙂 cat\n\n", "zzz"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_decode_never_raises_on_broken_bytes() -> None:
    from tlab.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer()
    tokenizer.train("abc", vocab_size=256)

    # 0xE4 是一个汉字 UTF-8 编码的第一个字节，单独出现时不是合法字符
    # 生成时可能出现这种情况，解码应替换为 �，而不是抛异常
    assert tokenizer.decode([0xE4]) == "�"
