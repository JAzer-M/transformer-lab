from pathlib import Path

# 保存加载与 HF 对照（agent 编写）；BPETokenizer 本身由用户手写

TEXT = "Once upon a time, there was a little cat. The cat sat on the mat.\n" * 20


def trained_tokenizer(vocab_size: int = 256 + 40):
    from tlab.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer()
    tokenizer.train(TEXT, vocab_size)
    return tokenizer


def test_save_and_load_restore_merges_vocab_and_encoding(tmp_path: Path) -> None:
    from tlab.tokenizer.bpe_io import load_bpe, save_bpe

    tokenizer = trained_tokenizer()
    path = tmp_path / "bpe.json"

    save_bpe(tokenizer, path)
    loaded = load_bpe(path)

    # 合并表（含先后顺序）、词表、编码结果都与保存前一致
    assert list(loaded.merges.items()) == list(tokenizer.merges.items())
    assert loaded.vocab == tokenizer.vocab
    sample = "The little cat ran."
    assert loaded.encode(sample) == tokenizer.encode(sample)


def test_hf_comparison_reports_round_trip_and_compression() -> None:
    from tlab.tokenizer.hf_compare import compare_with_hf

    result = compare_with_hf(TEXT, "A cat sat. Once upon a mat.\n", vocab_size=256 + 40)

    # 两边都能无损还原；压缩率为 “字节数 / token 数”，都应大于 1（确实压缩了）
    assert result["ours_round_trip"] and result["hf_round_trip"]
    assert result["ours_bytes_per_token"] > 1
    assert result["hf_bytes_per_token"] > 1
    assert result["hf_vocab_size"] <= 256 + 40
    # 相对差 = |ours - hf| / hf
    expected = abs(result["ours_bytes_per_token"] - result["hf_bytes_per_token"])
    expected /= result["hf_bytes_per_token"]
    assert abs(result["relative_diff"] - expected) < 1e-9
