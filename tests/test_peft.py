"""F4 的 PEFT 对照（agent 编写）：名字对应、同样的 A 和 B 输出相同、几步训练和自研相同。

peft 只出现在 scripts/ 和 tests/（AGENTS 硬性规则 1）。用 conftest.py 里的小模型，CPU、fp32。
"""

import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from conftest import TINY_EOS

from tlab.convert.qwen3 import load_qwen3
from tlab.finetune.lora import LoRALinear
from tlab.finetune.lora_model import apply_lora, lora_weights
from tlab.finetune.sft import SFTConfig, train_lora
from tlab.finetune.sft_labels import build_example

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def peft_script() -> ModuleType:
    return load_script("train_peft")


def tiny_config(out_dir: Path, **fields: object) -> SFTConfig:
    values: dict = {
        "out_dir": str(out_dir),
        "r": 4,
        "alpha": 8.0,
        "batch_size": 2,
        "grad_accum_steps": 2,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "warmup_iters": 2,
        "eval_interval": 2,
        "eos_token_id": TINY_EOS,
        "device": "cpu",
    }
    values.update(fields)
    return SFTConfig(**values)


def our_model(tiny_model_dir: Path, b_std: float = 0.0) -> torch.nn.Module:
    model = load_qwen3(tiny_model_dir, torch.device("cpu"))
    torch.manual_seed(42)
    apply_lora(model, r=4, alpha=8)
    if b_std:
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(std=b_std)
    return model


def items(n: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """提示 [a, b]，答案是两个 b；token 都在小分词器的词表里（2–96）。"""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        a, b = torch.randint(2, 97, (2,), generator=g).tolist()
        out.append(build_example([a, b], [b, b], TINY_EOS))
    return out


# ---- 名字对应 ----


def test_every_lora_weight_has_exactly_one_peft_name(
    peft_script: ModuleType, tiny_model_dir: Path, tmp_path: Path
) -> None:
    ours = our_model(tiny_model_dir)
    peft_model = peft_script.build_peft_model(
        tiny_model_dir, tiny_config(tmp_path), torch.device("cpu")
    )

    mapped = {peft_script.peft_name(name) for name in lora_weights(ours)}
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}

    assert mapped == trainable
    # 例子：我们的 out_proj 在 HF 里叫 o_proj
    assert peft_script.peft_name("blocks.1.attn.out_proj.lora_B") == (
        "base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weight"
    )


@pytest.mark.parametrize(
    "name", ["blocks.0.mlp.gate_proj.lora_A", "blocks.0.attn.q_proj.weight", "lora_A"]
)
def test_unknown_names_are_an_error(peft_script: ModuleType, name: str) -> None:
    with pytest.raises(ValueError, match="名字"):
        peft_script.peft_name(name)


def test_peft_trains_only_lora_with_no_dropout(
    peft_script: ModuleType, tiny_model_dir: Path, tmp_path: Path
) -> None:
    ours = our_model(tiny_model_dir)
    peft_model = peft_script.build_peft_model(
        tiny_model_dir, tiny_config(tmp_path), torch.device("cpu")
    )

    count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert count == sum(p.numel() for p in ours.parameters() if p.requires_grad)
    settings = peft_model.peft_config["default"]
    assert settings.lora_dropout == 0.0
    assert (settings.r, settings.lora_alpha) == (4, 8.0)
    assert set(settings.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    # 和 M4 对齐时一样用 eager 注意力：sdpa 的数值和我们手写的注意力不同
    assert peft_model.base_model.model.config._attn_implementation == "eager"


# ---- 同样的 A、B，输出相同 ----


@torch.no_grad()
def test_same_lora_weights_give_the_same_output(
    peft_script: ModuleType, tiny_model_dir: Path, tmp_path: Path
) -> None:
    # B 不是 0：B 是 0 时旁路输出是 0，名字对错、缩放对错都看不出来
    ours = our_model(tiny_model_dir, b_std=0.5)
    peft_model = peft_script.build_peft_model(
        tiny_model_dir, tiny_config(tmp_path), torch.device("cpu")
    )
    weights = {n: p.detach() for n, p in lora_weights(ours).items()}

    peft_script.copy_into_peft(peft_model, weights)

    idx = torch.randint(2, 97, (2, 9), generator=torch.Generator().manual_seed(1))
    theirs = peft_script.LogitsOnly(peft_model)(idx)  # (B, T, V)
    torch.testing.assert_close(theirs, ours(idx), rtol=1e-5, atol=1e-5)
    # 再导出成我们的名字，应该和拷进去的一样
    exported = peft_script.export_from_peft(peft_model)
    assert exported.keys() == weights.keys()
    assert all(torch.equal(exported[k], weights[k]) for k in weights)


def test_copy_with_missing_weights_is_an_error(
    peft_script: ModuleType, tiny_model_dir: Path, tmp_path: Path
) -> None:
    ours = our_model(tiny_model_dir)
    peft_model = peft_script.build_peft_model(
        tiny_model_dir, tiny_config(tmp_path), torch.device("cpu")
    )
    weights = dict(lora_weights(ours))
    weights.pop("blocks.1.attn.v_proj.lora_B")

    with pytest.raises(ValueError, match="v_proj"):
        peft_script.copy_into_peft(peft_model, weights)


def test_copy_with_extra_weights_is_an_error(
    peft_script: ModuleType, tiny_model_dir: Path, tmp_path: Path
) -> None:
    ours = our_model(tiny_model_dir)
    peft_model = peft_script.build_peft_model(
        tiny_model_dir, tiny_config(tmp_path), torch.device("cpu")
    )
    weights = dict(lora_weights(ours))
    weights["blocks.5.attn.q_proj.lora_A"] = torch.zeros(4, 32)  # 小模型只有 2 层

    with pytest.raises(ValueError, match="layers.5"):
        peft_script.copy_into_peft(peft_model, weights)


# ---- 几步训练：和自研的结果相同 ----


def read_losses(out_dir: Path, column: str) -> list[float]:
    with open(out_dir / "metrics.csv", newline="") as f:
        return [float(r[column]) for r in csv.DictReader(f) if r[column] != ""]


def test_a_few_steps_with_peft_match_our_lora(
    peft_script: ModuleType, tiny_model_dir: Path, tmp_path: Path
) -> None:
    train, val = items(16), items(4, seed=1)
    device = torch.device("cpu")

    ours = our_model(tiny_model_dir)
    train_lora(ours, train, val, tiny_config(tmp_path / "ours"), device)
    init = torch.load(tmp_path / "ours" / "lora_init.pt", weights_only=True)

    peft_model = peft_script.build_peft_model(
        tiny_model_dir, tiny_config(tmp_path / "peft"), device
    )
    peft_script.copy_into_peft(peft_model, init["weights"])
    peft_script.train_peft(
        peft_model, train, val, tiny_config(tmp_path / "peft"), device, extra_info={}
    )

    # 同样的起点、数据、顺序和训练循环：每一步的 loss 只差浮点舍入
    for column in ("train_loss", "val_loss"):
        a = read_losses(tmp_path / "ours", column)
        b = read_losses(tmp_path / "peft", column)
        assert len(a) == len(b) > 0
        assert a == pytest.approx(b, abs=1e-5)
    final_ours = torch.load(tmp_path / "ours" / "lora.pt", weights_only=True)["weights"]
    final_peft = torch.load(tmp_path / "peft" / "lora.pt", weights_only=True)["weights"]
    for name, weight in final_ours.items():
        torch.testing.assert_close(final_peft[name], weight, rtol=1e-4, atol=1e-5)


# ---- 脚本 ----


def test_script_runs_from_the_given_start_and_records_provenance(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    from tlab.finetune.lora_model import load_lora, save_lora

    start = tmp_path / "start.pt"
    save_lora(our_model(tiny_model_dir, b_std=0.1), start)

    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_peft.py"),
            "--config",
            str(sft_config),
            "--model-dir",
            str(tiny_model_dir),
            f"init_lora={start}",
            f"out_dir={tmp_path / 'peft'}",
            "max_steps=2",
        ],
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
    info = json.loads((tmp_path / "peft" / "run.json").read_text(encoding="utf-8"))
    assert info["implementation"] == "peft"
    assert re.fullmatch(r"\d+\.\d+\.\d+", info["peft_version"])
    assert info["init_lora"]["path"] == str(start)
    assert info["init_lora"]["sha256"] == hashlib.sha256(start.read_bytes()).hexdigest()
    assert info["lora"] == {
        "r": 4,
        "alpha": 8.0,
        "targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
    }
    assert info["steps_run"] == 2
    assert re.fullmatch(r"\d+\.\d+\.\d+", info["transformers_version"])
    with open(tmp_path / "peft" / "metrics.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    # 同一个目录再跑一次：报错，不覆盖上一次的结果
    again = run_peft_script(
        sft_config, tiny_model_dir, f"init_lora={start}", f"out_dir={tmp_path / 'peft'}"
    )
    assert again.returncode != 0
    assert "out_dir" in again.stderr
    # 存档用的是我们的名字：可以直接装进自研模型，用同一个评测脚本评测
    model = load_qwen3(tiny_model_dir, torch.device("cpu"))
    assert load_lora(model, tmp_path / "peft" / "lora.pt")["r"] == 4


def test_script_requires_a_start(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_peft.py"),
            "--config",
            str(sft_config),
            "--model-dir",
            str(tiny_model_dir),
        ],
        capture_output=True,
        text=True,
    )

    # 没有共同的起点，两条曲线本来就不会重合，比较没有意义
    assert done.returncode != 0
    assert "需要 init_lora" in done.stderr


def test_src_does_not_depend_on_peft() -> None:
    hits = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if re.search(
            r"^\s*(import|from)\s+peft\b", path.read_text(encoding="utf-8"), re.M
        )
    ]

    assert hits == []


# ---- 比较曲线 ----

BASE_CONFIG = {"r": 8, "alpha": 16.0, "learning_rate": 2e-4, "seed": 1337}


def make_run(
    run_dir: Path,
    train: list[float],
    val: list[str],
    config: dict | None = None,
    init_from: Path | None = None,
) -> Path:
    """一个训练输出目录：metrics.csv、run.json，以及 lora_init.pt（自研的那次）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "grad_norm",
                "tokens",
                "tokens_per_second",
            ]
        )
        for i, (t, v) in enumerate(zip(train, val, strict=True)):
            writer.writerow([i + 1, t, v, 1e-4, 1.0, 100, 1000.0])
    info: dict = {"config": {**BASE_CONFIG, "out_dir": str(run_dir), **(config or {})}}
    if init_from is None:
        (run_dir / "lora_init.pt").write_bytes(b"start of " + run_dir.name.encode())
        info["init_lora"] = None
    else:
        info["config"]["init_lora"] = str(init_from)
        info["init_lora"] = {
            "path": str(init_from),
            "sha256": hashlib.sha256(init_from.read_bytes()).hexdigest(),
        }
    (run_dir / "run.json").write_text(json.dumps(info), encoding="utf-8")
    return run_dir


def run_compare(
    tmp_path: Path, *extra: str
) -> tuple[subprocess.CompletedProcess, dict]:
    out = tmp_path / "compare.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_curves.py"),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return done, report


def pair(
    tmp_path: Path, ours_train, ours_val, theirs_train, theirs_val, **theirs_config
):
    ours = make_run(tmp_path / "ours", ours_train, ours_val)
    theirs = make_run(
        tmp_path / "theirs",
        theirs_train,
        theirs_val,
        config=theirs_config,
        init_from=ours / "lora_init.pt",
    )
    return ours, theirs


def test_compare_reports_step_and_val_differences(tmp_path: Path) -> None:
    ours, theirs = pair(
        tmp_path,
        [1.0, 0.9, 0.8],
        ["", "0.95", "0.85"],
        [1.0, 0.905, 0.795],
        ["", "0.952", "0.853"],
    )
    seed = make_run(
        tmp_path / "seed", [1.1, 0.95, 0.82], ["", "0.97", "0.88"], {"seed": 2024}
    )

    done, report = run_compare(
        tmp_path, "--ours", str(ours), "--theirs", str(theirs), "--reference", str(seed)
    )

    assert done.returncode == 0, done.stderr
    assert report["steps"] == 3
    # 每一步的差：0、0.005、0.005；第 1 步 B 还是 0，只差基座
    assert report["first_step_train_diff"] == 0.0
    assert report["mean_abs_train_diff"] == pytest.approx(0.01 / 3)
    assert report["max_abs_train_diff"] == pytest.approx(0.005)
    assert [v["step"] for v in report["val"]] == [2, 3]
    assert report["final_val_diff"] == pytest.approx(0.003)
    # 参照：同一个实现换种子，val loss 差 0.02、0.03
    assert [v["diff"] for v in report["seed_gap_val"]] == pytest.approx([0.02, 0.03])
    assert report["seed_gap_final_val"] == pytest.approx(0.03)
    assert report["thresholds"] == {"mean_abs_train_diff": 0.01, "final_val_diff": 0.01}
    assert report["passed"] is True


def test_compare_fails_when_the_curves_drift_apart(tmp_path: Path) -> None:
    ours, theirs = pair(
        tmp_path, [1.0, 0.9, 0.8], ["", "", "0.85"], [1.0, 0.95, 0.85], ["", "", "0.9"]
    )

    done, report = run_compare(tmp_path, "--ours", str(ours), "--theirs", str(theirs))

    # 平均差 0.0333、最后的 val 差 0.05，都超过 0.01：退出码不是 0，结果照样写出来
    assert done.returncode != 0
    assert report["passed"] is False
    assert "seed_gap_final_val" not in report


def test_compare_needs_both_conditions(tmp_path: Path) -> None:
    # 每一步都几乎一样，只有最后的 val loss 差 0.02：也算不达标
    ours, theirs = pair(
        tmp_path, [1.0, 0.9, 0.8], ["", "", "0.85"], [1.0, 0.9, 0.8], ["", "", "0.87"]
    )

    done, report = run_compare(tmp_path, "--ours", str(ours), "--theirs", str(theirs))

    assert report["mean_abs_train_diff"] == 0.0
    assert report["passed"] is False
    assert done.returncode != 0


def test_compare_refuses_runs_of_different_length(tmp_path: Path) -> None:
    # 看 val loss 的步数相同（都在第 2 步），但一次走了 3 步、一次只走了 2 步
    ours, theirs = pair(
        tmp_path, [1.0, 0.9, 0.8], ["", "0.86", ""], [1.0, 0.9], ["", "0.86"]
    )

    done, _ = run_compare(tmp_path, "--ours", str(ours), "--theirs", str(theirs))

    assert done.returncode != 0
    assert "3 步和 2 步" in done.stderr


def test_compare_refuses_a_reference_without_the_last_val_loss(tmp_path: Path) -> None:
    ours, theirs = pair(tmp_path, [1.0, 0.9], ["", "0.85"], [1.0, 0.9], ["", "0.85"])
    seed = make_run(tmp_path / "seed", [1.0, 0.9], ["0.9", ""], {"seed": 2024})

    done, _ = run_compare(
        tmp_path, "--ours", str(ours), "--theirs", str(theirs), "--reference", str(seed)
    )

    assert done.returncode != 0
    assert "参照" in done.stderr


def test_compare_refuses_runs_from_different_starts(tmp_path: Path) -> None:
    ours = make_run(tmp_path / "ours", [1.0, 0.9], ["", "0.85"])
    other = make_run(tmp_path / "other", [1.0, 0.9], ["", "0.85"])
    theirs = make_run(
        tmp_path / "theirs", [1.0, 0.9], ["", "0.85"], init_from=other / "lora_init.pt"
    )

    done, _ = run_compare(tmp_path, "--ours", str(ours), "--theirs", str(theirs))

    # 起点不同，曲线重合也说明不了什么
    assert done.returncode != 0
    assert "起点不同" in done.stderr


def test_compare_refuses_runs_with_different_settings(tmp_path: Path) -> None:
    ours, theirs = pair(
        tmp_path, [1.0, 0.9], ["", "0.85"], [1.0, 0.9], ["", "0.85"], learning_rate=1e-4
    )

    done, _ = run_compare(tmp_path, "--ours", str(ours), "--theirs", str(theirs))

    assert done.returncode != 0
    assert "learning_rate" in done.stderr


def test_f4_config_differs_from_f2_r8_only_in_start_and_out_dir() -> None:
    from dataclasses import replace

    from tlab.finetune.sft import load_sft_config

    f2 = load_sft_config(ROOT / "configs" / "finetune" / "f2_r8.yaml", ["seed=1337"])
    f4 = load_sft_config(ROOT / "configs" / "finetune" / "f4_peft_r8.yaml")

    assert f4.init_lora == "runs/m5/f2_r8_seed1337/lora_init.pt"
    assert f4.out_dir == "runs/m5/f4_peft_r8_seed1337"
    assert replace(f4, init_lora=None, out_dir=f2.out_dir) == f2


def run_peft_script(sft_config: Path, tiny_model_dir: Path, *overrides: str):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_peft.py"),
            "--config",
            str(sft_config),
            "--model-dir",
            str(tiny_model_dir),
            *overrides,
        ],
        capture_output=True,
        text=True,
    )


def test_script_refuses_a_bf16_base(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    from tlab.finetune.lora_model import save_lora

    start = tmp_path / "start.pt"
    save_lora(our_model(tiny_model_dir), start)

    done = run_peft_script(
        sft_config, tiny_model_dir, f"init_lora={start}", "base_dtype=bfloat16"
    )

    # 自研那次是 fp32 的基座，PEFT 也得一样，否则比的就不只是 LoRA 了
    assert done.returncode != 0
    assert "float32" in done.stderr


def test_script_refuses_a_start_with_other_settings(
    tmp_path: Path, sft_config: Path, tiny_model_dir: Path
) -> None:
    from tlab.finetune.lora_model import save_lora

    model = load_qwen3(tiny_model_dir, torch.device("cpu"))
    apply_lora(model, r=2, alpha=8)  # 配置里是 r=4
    start = tmp_path / "start_r2.pt"
    save_lora(model, start)

    done = run_peft_script(sft_config, tiny_model_dir, f"init_lora={start}")

    assert done.returncode != 0
    assert "init_lora 的设置" in done.stderr
