from dataclasses import asdict
from pathlib import Path

import pytest

# M3 消融配置的单变量规则：每个变体与对照组 a0_control 相比，只允许差下表里的字段
# （out_dir 按变体名区分，单独检查）。变体名见 docs/PLAN.md「M3 实施计划」。

ABLATIONS = Path(__file__).resolve().parents[1] / "configs" / "ablations"

HIGH_LR = {"learning_rate": 3e-3, "min_lr": 3e-4}
EXPECTED_DIFFS = {
    "a2_postln": {"norm_position": "post"},
    "a2_rmsnorm": {"norm": "rmsnorm"},
    "a2_preln_lr3e-3": HIGH_LR,
    "a2_postln_lr3e-3": {"norm_position": "post", **HIGH_LR},
    "a2_rmsnorm_lr3e-3": {"norm": "rmsnorm", **HIGH_LR},
    "a3_swiglu": {"mlp": "swiglu"},
    "a1_sinusoidal": {"pos_encoding": "sinusoidal"},
    "a1_rope": {"pos_encoding": "rope"},
    "a4_gqa2": {"n_kv_heads": 2},
    "a4_mqa": {"n_kv_heads": 1},
    # 组合只作参考，不算单变量结论
    "a9_modern": {
        "pos_encoding": "rope",
        "norm": "rmsnorm",
        "mlp": "swiglu",
        "n_kv_heads": 2,
    },
}

# 按 TinyStories 的 8192 词表手算：V·C + 512·C + 6 层 × 每层参数 + 最终归一化
#   对照（LayerNorm、Pre）：每层 12C² + 4C，ln_f 2C → 13,969,152
#   Post-LN：没有 ln_f → 少 2C = 768 → 13,968,384
#   RMSNorm：13 个归一化各少 C = 384 → 少 4,992 → 13,964,160
#   SwiGLU：前馈 3·C·1024 = 1,179,648，与 GELU MLP 的 8C² 相同 → 13,969,152
#   正弦、RoPE：没有位置表 → 少 512·C = 196,608 → 13,772,544
#   GQA-2：k、v 从 C² 降到 C·2·64，每层少 196,608，6 层少 1,179,648 → 12,789,504
#   MQA：k、v 降到 C·64，每层少 245,760，6 层少 1,474,560 → 12,494,592
#   组合：没有位置表、RMSNorm、GQA-2（SwiGLU 不变）→ 12,587,904
EXPECTED_PARAMS = {
    "a0_control": 13_969_152,
    "a2_postln": 13_968_384,
    "a2_rmsnorm": 13_964_160,
    "a2_preln_lr3e-3": 13_969_152,
    "a2_postln_lr3e-3": 13_968_384,
    "a2_rmsnorm_lr3e-3": 13_964_160,
    "a3_swiglu": 13_969_152,
    "a1_sinusoidal": 13_772_544,
    "a1_rope": 13_772_544,
    "a4_gqa2": 12_789_504,
    "a4_mqa": 12_494_592,
    "a9_modern": 12_587_904,
}


def load(variant: str) -> dict:
    from tlab.train.config import load_config

    return asdict(load_config(ABLATIONS / f"{variant}.yaml"))


@pytest.mark.parametrize("variant", sorted(EXPECTED_DIFFS))
def test_variant_differs_from_control_only_in_its_own_fields(variant: str) -> None:
    control = load("a0_control")
    config = load(variant)

    diffs = {
        key: config[key]
        for key in config
        if key != "out_dir" and config[key] != control[key]
    }

    assert diffs == EXPECTED_DIFFS[variant]
    assert config["out_dir"] == f"runs/m3/{variant}/seed1337"


@pytest.mark.parametrize("variant", sorted(EXPECTED_PARAMS))
def test_variant_parameter_count(variant: str) -> None:
    from tlab.model.gpt import GPT, count_parameters
    from tlab.train.utils import gpt_config_from

    model = GPT(gpt_config_from(load(variant), vocab_size=8192))

    assert count_parameters(model) == EXPECTED_PARAMS[variant]
