import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# 待实现模块在各测试内导入，与其他测试文件保持一致

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides):
    from tlab.train.config import TrainConfig

    # CPU 上很快跑完的极小配置：2 层，C=24，6 个头，d=4，窗口 16
    fields = {
        "n_layers": 2,
        "n_heads": 6,
        "d_model": 24,
        "block_size": 16,
        "batch_size": 2,
        "device": "cpu",
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def tiny_model(config, vocab_size: int = 11) -> torch.nn.Module:
    from dataclasses import asdict

    from tlab.model.gpt import GPT
    from tlab.train.utils import gpt_config_from

    torch.manual_seed(0)
    return GPT(gpt_config_from(asdict(config), vocab_size))


def test_median_of_runs_takes_the_middle_value_and_keeps_every_run() -> None:
    from tlab.eval.bench import median_of_runs

    runs = iter(
        [
            {"tokens_per_sec": 3.0, "peak_memory_mb": 30.0},
            {"tokens_per_sec": 1.0, "peak_memory_mb": 10.0},
            {"tokens_per_sec": 2.0, "peak_memory_mb": 50.0},
        ]
    )

    result = median_of_runs(lambda: next(runs), 3)

    assert result["tokens_per_sec"] == 2.0
    assert result["peak_memory_mb"] == 30.0
    assert [r["tokens_per_sec"] for r in result["runs"]] == [3.0, 1.0, 2.0]


def test_median_of_runs_keeps_none_when_memory_is_not_measured() -> None:
    from tlab.eval.bench import median_of_runs

    # CPU、MPS 上没有显存峰值
    result = median_of_runs(lambda: {"tokens_per_sec": 1.0, "peak_memory_mb": None}, 3)

    assert result["peak_memory_mb"] is None


@pytest.mark.parametrize(
    ("n_kv_heads", "expected_bytes"), [(None, 6144), (2, 2048), (1, 1024)]
)
def test_full_cache_size_matches_the_hand_calculation(
    n_kv_heads: int | None, expected_bytes: int
) -> None:
    from tlab.eval.bench import cache_bytes, fill_cache

    # 手算：2 层 × 2（k、v）× B=1 × g 组 × T=16 × d=4 × 每个数 4 字节（CPU 上是 fp32）
    # g=6 时是 6144 字节；g=2 是 1/3；g=1 是 1/6
    model = tiny_model(tiny_config(n_kv_heads=n_kv_heads))

    cache = fill_cache(model, 11, 16, torch.device("cpu"))

    assert cache_bytes(cache) == expected_bytes


@pytest.mark.parametrize(
    ("variant", "expected_mib", "k_dtype"),
    [
        ("a0_control", 4.5, torch.bfloat16),
        ("a4_gqa2", 1.5, torch.bfloat16),
        ("a4_mqa", 0.75, torch.bfloat16),
        # RoPE 的 cos、sin 是 fp32，bf16 的 k 旋转后被提升成 fp32，v 不旋转还是 bf16
        ("a1_rope", 6.75, torch.float32),
        ("a9_modern", 2.25, torch.float32),
    ],
)
def test_full_cache_of_real_configs_under_bf16_matches_the_runbook(
    variant: str, expected_mib: float, k_dtype: torch.dtype
) -> None:
    from tlab.eval.bench import cache_bytes, fill_cache
    from tlab.train.config import load_config

    # 手算：6 层 × g 组 × T=512 × d=64 × (k 的字节数 + v 的字节数)；Codex 按这些数字核对
    config = load_config(ROOT / "configs" / "ablations" / f"{variant}.yaml")
    model = tiny_model(config, vocab_size=8192)

    # CPU 上用 bf16 autocast 模拟 GPU 的训练精度
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cache = fill_cache(model, 8192, 512, torch.device("cpu"))

    assert cache_bytes(cache) / 2**20 == expected_mib
    assert cache[0].k.dtype == k_dtype
    assert cache[0].v.dtype == torch.bfloat16


@pytest.mark.parametrize(("grad_clip", "expected_clips"), [(0.0, 0), (1.0, 4)])
def test_training_measure_follows_the_real_training_step(
    monkeypatch: pytest.MonkeyPatch, grad_clip: float, expected_clips: int
) -> None:
    from tlab.eval import bench

    config = tiny_config(grad_accum_steps=2, grad_clip=grad_clip)
    model = tiny_model(config)
    forwards: list[tuple[int, ...]] = []
    model.register_forward_pre_hook(lambda _, args: forwards.append(args[0].shape))
    clips: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def counting_clip(parameters, max_norm):
        clips.append(max_norm)
        return real_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counting_clip)
    # 假时钟：计时开始在 10 秒，结束在 12 秒
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))

    result = bench.measure_training(model, config, 11, 3, torch.device("cpu"))

    # 第一步不计时，后面 3 步计时；每步累积 2 个 micro batch
    assert forwards == [(2, 16)] * 8
    assert len(clips) == expected_clips
    # 只算计时的 3 步：3 × 2 × B=2 × T=16 = 192 个 token，用了 2 秒
    assert result["tokens_per_sec"] == 96.0
    # 跑完释放梯度，不算进后面生成的显存峰值
    assert all(p.grad is None for p in model.parameters())


def test_generation_measure_counts_only_the_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tlab.eval import bench

    model = tiny_model(tiny_config())
    prompt = torch.zeros(2, 4, dtype=torch.long)  # (B, T)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))

    result = bench.measure_generation(model, prompt, 5, True, torch.device("cpu"))

    # B=2 × 5 个新 token = 10 个，用了 2 秒
    assert result["tokens_per_sec"] == 5.0


def test_bench_variant_reports_every_item(monkeypatch: pytest.MonkeyPatch) -> None:
    from tlab.eval import bench

    # 没有 GPU 也能确认每一次测量都记了温度
    monkeypatch.setattr(bench, "gpu_temperature_c", lambda: 50.0)

    result = bench.bench_variant(
        tiny_config(n_kv_heads=2),
        vocab_size=11,
        device=torch.device("cpu"),
        repeats=3,
        train_steps=2,
        prompt_len=4,
        new_tokens=4,
    )

    # 短测用的是配置里的结构
    assert result["config"]["n_kv_heads"] == 2
    assert result["device"] == "cpu"
    assert result["repeats"] == 3
    for item in ("train", "generate_no_cache", "generate_cache"):
        assert result[item]["tokens_per_sec"] > 0
        assert len(result[item]["runs"]) == 3
        assert result[item]["peak_memory_mb"] is None  # CPU 上没有显存峰值
        assert all(run["temperature_c"] == 50.0 for run in result[item]["runs"])
    assert result["cache_bytes_full_window"] == 2048
    # 记的是缓存里实际的类型，不是猜的
    assert result["cache_dtype"] == {"k": "torch.float32", "v": "torch.float32"}
    assert result["temperature_c"] == {"start": 50.0, "end": 50.0}


def test_bench_variant_rejects_a_generation_that_would_rebuild_the_cache() -> None:
    from tlab.eval.bench import bench_variant

    # 窗口满了以后缓存每一步都要重建，测出来的就不是缓存的速度
    with pytest.raises(ValueError, match="block_size"):
        bench_variant(
            tiny_config(),
            vocab_size=11,
            device=torch.device("cpu"),
            repeats=1,
            train_steps=1,
            prompt_len=8,
            new_tokens=9,
        )


def test_gpu_temperature_reads_nvidia_smi_and_is_none_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tlab.eval import bench

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="54\n")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    assert bench.gpu_temperature_c() == 54.0

    def no_tool(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(bench.subprocess, "run", no_tool)
    assert bench.gpu_temperature_c() is None


def test_script_writes_json_and_a_table_for_each_config(tmp_path: Path) -> None:
    config = tmp_path / "a4_tiny.yaml"
    config.write_text(
        "n_layers: 1\nn_heads: 2\nn_kv_heads: 1\nd_model: 8\nblock_size: 8\n"
        "batch_size: 2\n",
        encoding="utf-8",
    )
    out = tmp_path / "bench.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_attention.py"

    done = subprocess.run(
        [
            sys.executable,
            str(script),
            "--configs",
            str(config),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--vocab-size",
            "11",
            "--repeats",
            "1",
            "--train-steps",
            "1",
            "--prompt-len",
            "2",
            "--new-tokens",
            "2",
            "--cooldown",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
    results = json.loads(out.read_text(encoding="utf-8"))
    assert set(results) == {"a4_tiny"}
    # 1 层 × 2 × 1 组 × T=8 × d=4 × 4 字节 = 256
    assert results["a4_tiny"]["cache_bytes_full_window"] == 256
    # 表格写在 json 旁边，也打印出来
    table = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "a4_tiny" in table
    assert "a4_tiny" in done.stdout


def test_script_cools_down_before_every_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    # 第一个配置（对照组）紧跟在 6 次训练之后，不降温的话它测的时候最热
    config = "n_layers: 1\nn_heads: 2\nd_model: 8\nblock_size: 8\nbatch_size: 2\n"
    paths = [tmp_path / "first.yaml", tmp_path / "second.yaml"]
    for path in paths:
        path.write_text(config, encoding="utf-8")
    out = tmp_path / "bench.json"
    spec = importlib.util.spec_from_file_location(
        "bench_attention", ROOT / "scripts" / "bench_attention.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sleeps: list[float] = []
    monkeypatch.setattr(script, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_attention.py",
            "--configs",
            *map(str, paths),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--vocab-size",
            "11",
            "--repeats",
            "1",
            "--train-steps",
            "1",
            "--prompt-len",
            "2",
            "--new-tokens",
            "2",
            "--cooldown",
            "7",
        ],
    )

    script.main()

    assert sleeps == [7.0, 7.0]
    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["first"]["cooldown_seconds"] == 7.0
