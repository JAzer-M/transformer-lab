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
from conftest import TINY_EOS, write_data

from tlab.convert.qwen3 import load_qwen3
from tlab.finetune.lora import LoRALinear
from tlab.finetune.lora_model import apply_lora, save_lora
from tlab.finetune.sft import load_sft_config, run_sft

ROOT = Path(__file__).resolve().parents[1]


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


# ---- 从给定的 A、B 开始训练（PEFT 对照用同一个起点）----


def make_start_file(tmp_path: Path, tiny_model_dir: Path, r: int = 4) -> Path:
    """在小模型上装 LoRA，把 A、B 都设成随机值存下来，当作训练的起点。"""
    model = load_qwen3(tiny_model_dir, torch.device("cpu"))
    torch.manual_seed(42)
    apply_lora(model, r=r, alpha=8)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.1)
    path = tmp_path / f"start_r{r}.pt"
    save_lora(model, path)
    return path


def test_init_lora_starts_from_the_given_weights(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    start = make_start_file(tmp_path, tiny_model_dir)
    config = load_sft_config(sft_config, [f"init_lora={start}", "max_steps=1"])

    info = run_sft(config, tiny_model_dir)

    # 训练前存下的 A、B 就是给定的那一份，不是按种子随机出来的
    given = torch.load(start, weights_only=True)["weights"]
    used = torch.load(tmp_path / "run" / "lora_init.pt", weights_only=True)["weights"]
    assert given.keys() == used.keys()
    assert all(torch.equal(given[k], used[k]) for k in given)
    assert info["init_lora"]["path"] == str(start)
    assert len(info["init_lora"]["sha256"]) == 64


def test_init_lora_with_other_settings_is_an_error(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    start = make_start_file(tmp_path, tiny_model_dir, r=2)  # 配置里是 r=4
    config = load_sft_config(sft_config, [f"init_lora={start}"])

    with pytest.raises(ValueError, match="r"):
        run_sft(config, tiny_model_dir)
