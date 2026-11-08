"""几个测试文件共用的小模型。"""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM

TINY_EOS = 1  # 小分词器里 <|endoftext|> 的编号


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """结构和 Qwen3 相同的小模型，加一个词表 97 的小分词器（t2…t96 各是一个词）。"""
    path = tmp_path_factory.mktemp("tiny_qwen3_eval")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attn_implementation="eager",
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    vocab = {"[UNK]": 0, "<|endoftext|>": TINY_EOS}
    vocab |= {f"t{i}": i for i in range(2, 97)}
    tokenizer = Tokenizer(WordLevel(vocab, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    return path
