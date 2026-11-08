"""几个测试文件共用的小模型和小数据。"""

import json
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


def write_data(data_dir: Path, n_train: int = 40, n_test: int = 6) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", n_train), ("test", n_test)):
        rows = [
            json.dumps(
                {"question": f"t{2 + i % 9} ?", "answer": f"t{3 + i % 7} #### 5"}
            )
            for i in range(count)
        ]
        (data_dir / f"{split}.jsonl").write_text("\n".join(rows) + "\n", "utf-8")


@pytest.fixture()
def sft_config(tmp_path: Path) -> Path:
    """训练集 40 条：留 4 条验证，36 条训练；每步 2 × 2 = 4 条，共 9 步。"""
    data_dir = tmp_path / "data"
    write_data(data_dir)
    split = tmp_path / "split.json"
    val = [0, 10, 20, 30]
    split.write_text(
        json.dumps(
            {
                "seed": 7,
                "train_total": 40,
                "val_indices": val,
                "train_indices": [i for i in range(40) if i not in val],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "sft.yaml"
    config.write_text(
        f"data_dir: {data_dir}\nsplit_file: {split}\nout_dir: {tmp_path / 'run'}\n"
        f"r: 4\nalpha: 8\nbatch_size: 2\ngrad_accum_steps: 2\nwarmup_iters: 2\n"
        f"learning_rate: 1.0e-3\nmin_lr: 1.0e-4\neval_interval: 4\n"
        f"eos_token_id: {TINY_EOS}\ndevice: cpu\n",
        encoding="utf-8",
    )
    return config
