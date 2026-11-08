"""SFT 的入口（agent 编写）：读数据和划分、载入基座、训练、存档；评测时载入 LoRA。

用 conftest.py 里结构和 Qwen3 相同的小模型，不联网。
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import TINY_EOS

from tlab.convert.qwen3 import load_qwen3
from tlab.finetune.lora import LoRALinear
from tlab.finetune.lora_model import apply_lora, save_lora
from tlab.finetune.sft import load_sft_config, run_sft

ROOT = Path(__file__).resolve().parents[1]


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


def run_train(config: Path, model_dir: Path, *overrides: str):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sft.py"),
            "--config",
            str(config),
            "--model-dir",
            str(model_dir),
            *overrides,
        ],
        capture_output=True,
        text=True,
    )


def test_train_script_writes_metrics_run_info_and_lora(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    done = run_train(sft_config, tiny_model_dir)

    assert done.returncode == 0, done.stderr
    out_dir = tmp_path / "run"
    info = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert info["model"] == str(tiny_model_dir)
    assert info["n_train"] == 36 and info["n_val"] == 4
    assert info["dropped_train"] == info["dropped_val"] == 0
    assert info["split_seed"] == 7
    assert info["total_steps"] == info["steps_run"] == 9
    assert info["lora"] == {
        "r": 4,
        "alpha": 8.0,
        "targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
    }
    assert (out_dir / "lora.pt").exists()
    assert (out_dir / "metrics.csv").read_text(encoding="utf-8").count("\n") == 1 + 9
    assert "val" in done.stdout


def test_short_run_with_max_steps(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    done = run_train(sft_config, tiny_model_dir, "max_steps=2")

    assert done.returncode == 0, done.stderr
    info = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert info["steps_run"] == 2 and info["total_steps"] == 9


def test_same_seed_gives_the_same_lora(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    weights = []
    for name in ("a", "b"):
        config = load_sft_config(sft_config, [f"out_dir={tmp_path / name}"])
        run_sft(config, tiny_model_dir)
        weights.append(torch.load(tmp_path / name / "lora.pt", weights_only=True))

    assert weights[0]["weights"].keys() == weights[1]["weights"].keys()
    for key, value in weights[0]["weights"].items():
        assert torch.equal(value, weights[1]["weights"][key]), key


def test_split_that_does_not_match_the_data_is_an_error(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    write_data(tmp_path / "data", n_train=41)  # 划分文件写的是 40 条

    with pytest.raises(ValueError, match="40"):
        run_sft(load_sft_config(sft_config), tiny_model_dir)


def test_bfloat16_base_needs_a_gpu(sft_config: Path, tiny_model_dir: Path) -> None:
    # CPU 上没有 autocast：bf16 的基座和 fp32 的 A、B 相乘会报类型不一致
    config = load_sft_config(sft_config, ["base_dtype=bfloat16"])

    with pytest.raises(ValueError, match="GPU"):
        run_sft(config, tiny_model_dir)


# ---- 评测时载入 LoRA ----


def write_eval_config(tmp_path: Path, lora: Path | None) -> Path:
    data_dir = tmp_path / "data"
    write_data(data_dir)
    subset = tmp_path / "subset.json"
    subset.write_text(
        json.dumps(
            {
                "seed": 7,
                "test_total": 6,
                "test_indices": [1, 3],
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
    path = tmp_path / ("eval_lora.yaml" if lora else "eval_base.yaml")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_eval_script_loads_and_merges_a_lora(
    tmp_path: Path, tiny_model_dir: Path
) -> None:
    # 一个 B 很大的 LoRA：装上之后生成的内容一定会变
    model = load_qwen3(tiny_model_dir, torch.device("cpu"))
    torch.manual_seed(0)
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=1.0)
    lora_path = tmp_path / "lora.pt"
    save_lora(model, lora_path)

    outputs = {}
    for lora in (None, lora_path):
        out = tmp_path / ("with.json" if lora else "without.json")
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "eval_gsm8k.py"),
                "--config",
                str(write_eval_config(tmp_path, lora)),
                "--model-dir",
                str(tiny_model_dir),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        outputs[lora] = json.loads(out.read_text(encoding="utf-8"))

    assert outputs[None]["lora"] is None
    # 记下存档的哈希：同一个路径的存档被重新训练覆盖后，也分得清用的是哪一份
    sha256 = hashlib.sha256(lora_path.read_bytes()).hexdigest()
    assert outputs[lora_path]["lora"] == {
        "path": str(lora_path),
        "sha256": sha256,
        "r": 4,
        "alpha": 8.0,
        "targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
    }
    base_text = [item["output"] for item in outputs[None]["items"]]
    lora_text = [item["output"] for item in outputs[lora_path]["items"]]
    assert base_text != lora_text


def test_config_for_evaluating_the_sft_model() -> None:
    from tlab.eval.gsm8k import load_eval_config

    sft = load_eval_config(ROOT / "configs" / "finetune" / "f1_sft_r8.yaml")
    base = load_eval_config(ROOT / "configs" / "finetune" / "f1_base_zeroshot.yaml")

    # 和基座不给例子的评测只差 LoRA 存档：同样的题目、提示和生成设置
    # F1 里“微调后”的数字用 F2 里 r=8、种子 1337 那一次
    assert sft.lora == "runs/m5/f2_r8_seed1337/lora.pt"
    assert replace(sft, lora=None) == base
