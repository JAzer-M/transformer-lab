"""几个测试文件共用的小模型和小数据。"""

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
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


def load_script(name: str) -> ModuleType:
    """把 scripts/ 下的脚本当模块导入，测试里可以替换它用到的函数。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def big_lora_model(model_dir: Path) -> torch.nn.Module:
    """小 Qwen3 装上 r=4 的 LoRA，B 是很大的随机值：装上之后生成的内容一定会变。"""
    # 在函数里导入：lora.py 出错时只影响用到它的测试
    from tlab.convert.qwen3 import load_qwen3
    from tlab.finetune.lora import LoRALinear
    from tlab.finetune.lora_model import apply_lora

    model = load_qwen3(model_dir, torch.device("cpu"))
    torch.manual_seed(0)
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=1.0)
    return model


def write_eval_config(
    directory: Path, lora: Path | None, test_indices: Sequence[int] = (1, 3)
) -> Path:
    """在 directory 下写小数据、题目子集和一份不给例子的评测配置。"""
    data_dir = directory / "data"
    write_data(data_dir)
    subset = directory / "subset.json"
    subset.write_text(
        json.dumps(
            {
                "seed": 7,
                "test_total": 6,
                "test_indices": list(test_indices),
                "train_total": 40,
                "few_shot_indices": [0],
            }
        ),
        encoding="utf-8",
    )
    lines = [
        "prompt_mode: zero_shot",
        "max_new_tokens: 6",
        f"eos_token_id: {TINY_EOS}",
        f"data_dir: {data_dir}",
        f"subset_file: {subset}",
        "device: cpu",
    ]
    if lora is not None:
        lines.append(f"lora: {lora}")
    path = directory / ("eval_lora.yaml" if lora else "eval_base.yaml")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
