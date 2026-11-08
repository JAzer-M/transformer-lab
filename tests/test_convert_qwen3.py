import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

# 权重转换（M4）。transformers 只在这里当对照，src/tlab 不依赖它（AGENTS 硬性规则 1）。
# 用随机初始化的小模型，不需要下载；真实权重的验收是 scripts/check_qwen3.py

ROOT = Path(__file__).resolve().parents[1]

# Qwen3-0.6B 真实的 config.json（2026-11-07 从 HuggingFace 读到的）
REAL_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_qwen3", ROOT / "scripts" / "check_qwen3.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def tiny_hf_model(n_layers: int = 2) -> nn.Module:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    # 结构和 Qwen3-0.6B 相同，只是小很多：d=16 不等于 C/h=8。
    # 初始值放大一些，不然 logits 都接近 0，算错了误差也小，测试就没有意义
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        initializer_range=0.3,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).eval()
    # HF 把 RMSNorm 的 weight 初始化成全 1。全 1 时 q_norm、k_norm 或 ln1、ln2 写反，
    # 输出还是完全一样，查不出来；真实权重不是全 1，所以这里也打乱
    for name, parameter in model.named_parameters():
        if "norm" in name:
            torch.nn.init.normal_(parameter, mean=1.0, std=0.3)
    return model


@pytest.fixture(scope="module")
def tiny_hf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny_qwen3")
    tiny_hf_model().save_pretrained(path)
    return path


def hf_state(n_layers: int = 2) -> dict[str, torch.Tensor]:
    """HF 权重文件里的名字和形状（C=32、h=4、g=2、d=16、H=48、V=97），数值随机。"""
    torch.manual_seed(0)
    layer = {
        "input_layernorm.weight": (32,),
        "post_attention_layernorm.weight": (32,),
        "self_attn.q_proj.weight": (64, 32),
        "self_attn.k_proj.weight": (32, 32),
        "self_attn.v_proj.weight": (32, 32),
        "self_attn.o_proj.weight": (32, 64),
        "self_attn.q_norm.weight": (16,),
        "self_attn.k_norm.weight": (16,),
        "mlp.gate_proj.weight": (48, 32),
        "mlp.up_proj.weight": (48, 32),
        "mlp.down_proj.weight": (32, 48),
    }
    state = {
        "model.embed_tokens.weight": torch.randn(97, 32),
        "model.norm.weight": torch.randn(32),
    }
    for i in range(n_layers):
        for name, shape in layer.items():
            state[f"model.layers.{i}.{name}"] = torch.randn(*shape)
    return state


# ---- 读配置 ----


def test_real_config_gives_the_qwen3_structure() -> None:
    from tlab.convert.qwen3 import gpt_config_from_hf

    config = gpt_config_from_hf(REAL_CONFIG)

    assert config.vocab_size == 151_936
    assert config.block_size == 40_960
    assert config.d_model == 1024
    assert config.n_layers == 28
    assert config.n_heads == 16
    assert config.n_kv_heads == 8
    assert config.head_dim == 128
    assert config.qk_norm is True
    assert config.norm == "rmsnorm"
    assert config.norm_position == "pre"
    assert config.mlp == "swiglu"
    assert config.mlp_hidden_dim == 3072
    assert config.pos_encoding == "rope"
    assert config.rope_theta == 1_000_000.0


def test_config_saved_by_new_transformers_gives_the_same_rope_theta() -> None:
    from tlab.convert.qwen3 import gpt_config_from_hf

    # 新版 transformers 把 rope_theta 写在 rope_parameters 里
    new = {k: v for k, v in REAL_CONFIG.items() if k != "rope_theta"}
    new["rope_parameters"] = {"rope_theta": 1000000, "rope_type": "default"}

    assert gpt_config_from_hf(new).rope_theta == 1_000_000.0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_type", "llama"),
        ("attention_bias", True),
        ("hidden_act", "gelu"),
        ("rms_norm_eps", 1e-5),
        ("rope_scaling", {"rope_type": "linear"}),
        ("rope_parameters", {"rope_type": "linear", "rope_theta": 1000000}),
        ("attention_dropout", 0.1),
        ("tie_word_embeddings", False),
        ("use_sliding_window", True),
    ],
)
def test_config_rejects_what_our_structure_cannot_express(
    key: str, value: object
) -> None:
    from tlab.convert.qwen3 import gpt_config_from_hf

    # 不报错的话，权重能载入、结果却对不上，要很久才查得出来
    with pytest.raises(ValueError, match=key):
        gpt_config_from_hf({**REAL_CONFIG, key: value})


def test_head_dim_falls_back_to_c_over_h_like_the_official_code() -> None:
    from tlab.convert.qwen3 import gpt_config_from_hf

    # 官方代码：配置里没写 head_dim 时用 hidden_size // num_attention_heads
    without = {k: v for k, v in REAL_CONFIG.items() if k != "head_dim"}

    assert gpt_config_from_hf(without).head_dim == 64


# ---- 映射权重 ----


def test_every_hf_tensor_goes_to_exactly_one_parameter() -> None:
    from tlab.convert.qwen3 import convert_state_dict, gpt_config_from_hf
    from tlab.model.gpt import GPT

    state = hf_state()
    tiny = {
        **REAL_CONFIG,
        "vocab_size": 97,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
    }
    model = GPT(gpt_config_from_hf(tiny))

    converted = convert_state_dict(state, n_layers=2)

    # 严格载入：多一个或少一个名字都会报错
    model.load_state_dict(converted)
    # 抽几个核对对应关系，形状一样的 k、v 写反了只有这样才查得出来
    attn = model.blocks[1].attn
    pairs = [
        (model.tok_emb.weight, "model.embed_tokens.weight"),
        (model.ln_f.weight, "model.norm.weight"),
        (model.blocks[0].ln1.weight, "model.layers.0.input_layernorm.weight"),
        (model.blocks[0].ln2.weight, "model.layers.0.post_attention_layernorm.weight"),
        (attn.q_proj.weight, "model.layers.1.self_attn.q_proj.weight"),
        (attn.k_proj.weight, "model.layers.1.self_attn.k_proj.weight"),
        (attn.v_proj.weight, "model.layers.1.self_attn.v_proj.weight"),
        (attn.out_proj.weight, "model.layers.1.self_attn.o_proj.weight"),
        (attn.q_norm.weight, "model.layers.1.self_attn.q_norm.weight"),
        (attn.k_norm.weight, "model.layers.1.self_attn.k_norm.weight"),
        (model.blocks[1].mlp.gate_proj.weight, "model.layers.1.mlp.gate_proj.weight"),
        (model.blocks[1].mlp.up_proj.weight, "model.layers.1.mlp.up_proj.weight"),
        (model.blocks[1].mlp.down_proj.weight, "model.layers.1.mlp.down_proj.weight"),
    ]
    for parameter, name in pairs:
        assert torch.equal(parameter, state[name]), name
    # 权重共享还在：lm_head 和词表是同一个参数
    assert model.lm_head.weight is model.tok_emb.weight


def test_real_file_has_311_tensors_and_they_all_map() -> None:
    from tlab.convert.qwen3 import convert_state_dict

    # 真实文件：28 层 × 11 个，加词表、norm、lm_head 各 1 个
    state = {
        "model.embed_tokens.weight": torch.zeros(1),
        "model.norm.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }
    layer = hf_state(1)
    for i in range(28):
        for name in layer:
            if name.startswith("model.layers.0."):
                short = name.removeprefix("model.layers.0.")
                state[f"model.layers.{i}.{short}"] = torch.zeros(1)
    assert len(state) == 311

    converted = convert_state_dict(state, n_layers=28)

    # 我们的 state_dict 里 lm_head 和词表各有一个名字，所以数量还是 311
    assert len(converted) == 311


def test_unknown_tensor_is_an_error() -> None:
    from tlab.convert.qwen3 import convert_state_dict

    state = {**hf_state(), "model.layers.0.self_attn.extra.weight": torch.zeros(1)}

    with pytest.raises(ValueError, match="extra"):
        convert_state_dict(state, n_layers=2)


def test_missing_tensor_is_an_error() -> None:
    from tlab.convert.qwen3 import convert_state_dict

    state = hf_state()
    del state["model.layers.1.self_attn.k_norm.weight"]

    with pytest.raises(ValueError, match="k_norm"):
        convert_state_dict(state, n_layers=2)


def test_layer_beyond_n_layers_is_an_error() -> None:
    from tlab.convert.qwen3 import convert_state_dict

    # 文件有 2 层、配置只有 1 层，说明配置和权重不是一套
    with pytest.raises(ValueError, match="layers.1"):
        convert_state_dict(hf_state(2), n_layers=1)


def test_lm_head_must_equal_the_embeddings() -> None:
    from tlab.convert.qwen3 import convert_state_dict

    state = hf_state()
    # 文件里单独存了一份、数值相同，这是真实文件的情况
    same = {**state, "lm_head.weight": state["model.embed_tokens.weight"].clone()}
    convert_state_dict(same, n_layers=2)

    # 我们的结构是共享权重，不相同就表示不了这份权重
    different = {**state, "lm_head.weight": torch.randn(97, 32)}
    with pytest.raises(ValueError, match="lm_head"):
        convert_state_dict(different, n_layers=2)


def test_load_qwen3_reads_config_and_weights_from_a_directory(
    tiny_hf_dir: Path,
) -> None:
    from tlab.convert.qwen3 import load_qwen3

    model = load_qwen3(tiny_hf_dir, torch.device("cpu"))

    assert model.blocks[0].attn.q_proj.weight.shape == (64, 32)
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.tok_emb.weight.dtype == torch.float32


def test_load_qwen3_converts_bfloat16_weights_to_fp32(tmp_path: Path) -> None:
    from tlab.convert.qwen3 import load_qwen3

    # 真实文件是 bf16；存的时候转一下，载入后要回到 fp32，对齐才在 fp32 下比较
    tiny_hf_model().to(torch.bfloat16).save_pretrained(tmp_path)

    model = load_qwen3(tmp_path, torch.device("cpu"))

    assert model.tok_emb.weight.dtype == torch.float32
    assert model.blocks[0].attn.q_norm.weight.dtype == torch.float32


# ---- 和 transformers 对比 ----


def test_every_layer_and_the_logits_match_transformers(tiny_hf_dir: Path) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    device = torch.device("cpu")
    ours = load_qwen3(tiny_hf_dir, device)
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    ids = torch.randint(0, 97, (2, 12))  # (B, T)

    result = script.compare_layers(ours, theirs, ids)

    # 词表、2 层、最终归一化，一共 4 处
    assert len(result["layer_max_errors"]) == 4
    assert max(result["layer_max_errors"]) < 1e-4
    assert result["logits_max_error"] < 1e-4
    # 分数本身要够大，不然“误差小”没有意义
    assert result["logits_max_abs"] > 0.1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("blocks.1.attn.k_proj.weight", "blocks.1.attn.v_proj.weight"),
        ("blocks.1.attn.q_norm.weight", "blocks.1.attn.k_norm.weight"),
        ("blocks.1.ln1.weight", "blocks.1.ln2.weight"),
        ("blocks.1.mlp.gate_proj.weight", "blocks.1.mlp.up_proj.weight"),
    ],
)
def test_comparison_notices_swapped_weights_of_the_same_shape(
    tiny_hf_dir: Path, first: str, second: str
) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    # 形状相同的两个参数对调：严格载入看不出来，只有比较输出才知道
    state = ours.state_dict()
    a = state[first].clone()  # 形状和 second 相同
    state[first].copy_(state[second])
    state[second].copy_(a)

    result = script.compare_layers(ours, theirs, torch.randint(0, 97, (1, 12)))

    errors = result["layer_max_errors"]
    # 词表和第 1 层还对得上，从第 2 层开始对不上：逐层对比才能指出是哪一层
    assert errors[0] < 1e-4 and errors[1] < 1e-4
    assert errors[2] > 1e-3
    assert result["logits_max_error"] > 1e-3


def test_comparison_reports_relative_errors_of_hidden_states(tiny_hf_dir: Path) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()

    # 小模型两边几乎完全相同，误差是 0 就测不出除法，所以先改掉一个 weight，让误差不为 0
    torch.nn.init.normal_(ours.blocks[1].ln1.weight)

    result = script.compare_layers(ours, theirs, torch.randint(0, 97, (1, 12)))

    # 隐藏状态的数值可能上千，只看绝对误差说明不了什么，要除以它本身的大小
    assert len(result["layer_max_abs"]) == 4
    assert min(result["layer_max_abs"]) > 0
    assert result["layer_max_errors"][3] > 1e-3
    assert result["layer_relative_errors"] != result["layer_max_errors"]
    expected = [
        error / size
        for error, size in zip(
            result["layer_max_errors"], result["layer_max_abs"], strict=True
        )
    ]
    assert result["layer_relative_errors"] == expected


def test_generation_is_not_cut_short_by_the_eos_token(tiny_hf_dir: Path) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    ids = torch.randint(0, 97, (1, 5))  # (1, T)
    first = script.compare_generation(ours, theirs, ids, new_tokens=10)
    # 把第 1 个新 token 设成 eos：不关掉 eos 的话，transformers 生成一个就停了，
    # 长度和我们的不一样。真实模型自带 eos，小模型默认没有，所以这里要设上
    theirs.generation_config.eos_token_id = first["tokens"][5]

    result = script.compare_generation(ours, theirs, ids, new_tokens=10)

    assert len(result["tokens"]) == 15
    assert result["same_with_cache"] is True


def test_generation_ignores_the_sampling_settings_of_the_model(
    tiny_hf_dir: Path,
) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    # 真实模型的 generation_config 是抽样，还可能带重复惩罚；这些都要关掉才是贪心
    theirs.generation_config.do_sample = True
    theirs.generation_config.temperature = 0.6
    theirs.generation_config.top_k = 20
    theirs.generation_config.top_p = 0.9
    theirs.generation_config.repetition_penalty = 1.5
    ids = torch.randint(0, 97, (1, 5))  # (1, T)

    result = script.compare_generation(ours, theirs, ids, new_tokens=20)

    assert result["same_with_cache"] is True
    assert result["same_without_cache"] is True


def test_comparison_notices_a_wrong_weight(tiny_hf_dir: Path) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    # 故意把第 2 层的 k、v 投影对调
    attn = ours.blocks[1].attn
    k = attn.k_proj.weight.data.clone()
    attn.k_proj.weight.data.copy_(attn.v_proj.weight.data)
    attn.v_proj.weight.data.copy_(k)

    result = script.compare_layers(ours, theirs, torch.randint(0, 97, (1, 12)))

    errors = result["layer_max_errors"]
    # 词表和第 1 层还对得上，从第 2 层开始对不上：逐层对比才能指出是哪一层
    assert errors[0] < 1e-4 and errors[1] < 1e-4
    assert errors[2] > 1e-3
    assert result["logits_max_error"] > 1e-3


def test_greedy_generation_matches_transformers_with_and_without_cache(
    tiny_hf_dir: Path,
) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    ids = torch.randint(0, 97, (1, 5))  # (1, T)

    result = script.compare_generation(ours, theirs, ids, new_tokens=20)

    assert result["same_without_cache"] is True
    assert result["same_with_cache"] is True
    assert result["new_tokens"] == 20
    assert len(result["tokens"]) == 25


def test_generation_comparison_notices_different_tokens(tiny_hf_dir: Path) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    # 把我们的最终归一化改掉，分数就变了，生成的 token 不可能还一样
    torch.manual_seed(1)
    torch.nn.init.normal_(ours.ln_f.weight)
    ids = torch.randint(0, 97, (1, 5))  # (1, T)

    result = script.compare_generation(ours, theirs, ids, new_tokens=20)

    assert result["same_without_cache"] is False
    assert result["same_with_cache"] is False


# ---- 检查脚本 ----


def run_script(tmp_path: Path, model_dir: Path, *extra: str) -> tuple:
    out = tmp_path / "out" / "alignment.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_qwen3.py"),
            "--model-dir",
            str(model_dir),
            "--out",
            str(out),
            "--new-tokens",
            "5",
            "--device",
            "cpu",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    return done, out


def test_script_writes_a_report_with_errors_and_provenance(
    tmp_path: Path, tiny_hf_dir: Path
) -> None:
    done, out = run_script(tmp_path, tiny_hf_dir)

    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["max_error_allowed"] == 1e-3
    assert report["dtype"] == "torch.float32"
    assert report["n_params"] > 0
    assert report["device"] == "cpu"
    assert report["git_commit"]
    assert report["transformers_version"]
    # 传 --model-dir 时记这个目录；下载时记 仓库名@版本号，换一台机器也知道用的是哪份权重
    assert report["model"] == str(tiny_hf_dir)
    # 验收标准说的是分数：max_error 就是所有提示里分数的最大误差
    assert report["max_error"] == max(p["logits_max_error"] for p in report["prompts"])
    assert len(report["prompts"]) >= 2
    for prompt in report["prompts"]:
        assert prompt["logits_max_error"] < 1e-3
        assert prompt["same_with_cache"] is True
        assert prompt["same_without_cache"] is True
    assert "最大误差" in done.stdout


def test_script_fails_when_the_error_is_too_big(
    tmp_path: Path, tiny_hf_dir: Path
) -> None:
    # 把门槛设成 0，一定超过：确认超过时退出码不是 0，手册和验收才能靠它判断
    done, out = run_script(tmp_path, tiny_hf_dir, "--max-error", "0")

    assert done.returncode != 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("logits_error", "same", "expected"),
    [
        (1e-4, True, True),
        (2e-3, True, False),
        (1e-4, False, False),
    ],
)
def test_only_logits_and_generation_decide_the_result(
    logits_error: float, same: bool, expected: bool
) -> None:
    script = load_script()
    # 隐藏状态的误差比门槛大，但数值本身上千，相对误差很小：这不算失败
    prompts = [
        {
            "layer_max_errors": [0.0, 5e-2],
            "logits_max_error": logits_error,
            "same_with_cache": same,
            "same_without_cache": True,
        }
    ]

    worst, passed = script.judge(prompts, max_error=1e-3)

    assert worst == logits_error
    assert passed is expected


def test_script_uses_the_tokenizer_when_there_is_one(
    tmp_path: Path, tiny_hf_dir: Path
) -> None:
    import shutil

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    model_dir = tmp_path / "with_tokenizer"
    shutil.copytree(tiny_hf_dir, model_dir)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "Once": 1, "upon": 2, "<|endoftext|>": 3}, "[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))

    done, out = run_script(tmp_path, model_dir)

    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    # 用的是真实的提示，也写出了生成的文字
    assert report["prompts"][2]["prompt"].startswith("Once upon a time")
    assert all("generated_text" in prompt for prompt in report["prompts"])


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], "Qwen/Qwen3-0.6B"),  # 不传时还是 M4 验收用的模型
        (["--repo", "Qwen/Qwen3-0.6B-Base"], "Qwen/Qwen3-0.6B-Base"),
    ],
)
def test_script_downloads_the_repo_it_is_given(
    tmp_path: Path,
    tiny_hf_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    expected: str,
) -> None:
    script = load_script()
    asked: list[str] = []

    def fake_download(repo: str, allow_patterns: list[str]) -> str:
        asked.append(repo)
        return str(tiny_hf_dir)

    monkeypatch.setattr(script, "snapshot_download", fake_download)
    out = tmp_path / "alignment.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_qwen3.py", "--out", str(out), "--new-tokens", "3", *extra],
    )

    script.main()

    assert asked == [expected]
    report = json.loads(out.read_text(encoding="utf-8"))
    # 结果里记着用的是哪个模型、哪个版本
    assert report["model"] == f"{expected}@{tiny_hf_dir.name}"


def test_bf16_is_reported_but_does_not_decide_the_result(
    tmp_path: Path, tiny_hf_dir: Path
) -> None:
    done, out = run_script(tmp_path, tiny_hf_dir)
    assert done.returncode == 0, done.stderr
    fp32 = json.loads(out.read_text(encoding="utf-8"))

    # 门槛设得再小也不算失败：bf16 下本来就有舍入，这次只看差多少
    done, out = run_script(
        tmp_path, tiny_hf_dir, "--autocast", "bf16", "--max-error", "1e-12"
    )

    assert done.returncode == 0, done.stderr
    bf16 = json.loads(out.read_text(encoding="utf-8"))
    assert fp32["autocast"] is None
    assert fp32["passed"] is True
    assert bf16["autocast"] == "bf16"
    assert bf16["passed"] is None
    # 两边真的在 bf16 下算了：分数是 bf16。原来这里断言“误差比 fp32 大 10 倍”，
    # rmsnorm_fp32 打开后 bf16 下误差变成 0，这条不再成立；2026-11-08 用户同意改成检查类型
    for prompt in bf16["prompts"]:
        assert prompt["logits_dtype"] == "torch.bfloat16"
        assert prompt["their_logits_dtype"] == "torch.bfloat16"
    for prompt in fp32["prompts"]:
        assert prompt["logits_dtype"] == "torch.float32"
        assert prompt["their_logits_dtype"] == "torch.float32"
    assert "不判定" in done.stdout


def test_loading_qwen3_turns_on_rmsnorm_fp32() -> None:
    from tlab.convert.qwen3 import gpt_config_from_hf

    # 官方的 Qwen3RMSNorm 在内部先转 fp32，载入时要和它一致
    assert gpt_config_from_hf(REAL_CONFIG).rmsnorm_fp32 is True


@torch.no_grad()
def test_bf16_matches_transformers_bit_for_bit_with_rmsnorm_fp32(
    tiny_hf_dir: Path,
) -> None:
    from transformers import Qwen3ForCausalLM

    from tlab.convert.qwen3 import load_qwen3
    from tlab.model.layers import RMSNorm

    script = load_script()
    ours = load_qwen3(tiny_hf_dir, torch.device("cpu"))
    theirs = Qwen3ForCausalLM.from_pretrained(
        tiny_hf_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 97, (2, 12), generator=generator)  # (B, T)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        on = script.compare_layers(ours, theirs, ids)
        for module in ours.modules():
            if isinstance(module, RMSNorm):
                module.compute_fp32 = False
        off = script.compare_layers(ours, theirs, ids)

    # 打开时每一层都和官方逐位相同；关掉就有差别，说明差别正是来自这里。
    # 注意 autocast 下残差主干是 fp32，这里真正受影响的是 q_norm、k_norm（输入是 bf16）。
    # 这是短序列；真实模型上 83 个 token 时，注意力内部还有 bf16 舍入差别（2026-11-08 审查实测）
    assert on["logits_dtype"] == "torch.bfloat16"
    assert max(on["layer_max_errors"]) == 0.0
    assert on["logits_max_error"] == 0.0
    assert off["logits_max_error"] > 0.0


def test_script_reports_the_rmsnorm_setting(tmp_path: Path, tiny_hf_dir: Path) -> None:
    done, out = run_script(tmp_path, tiny_hf_dir)

    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["rmsnorm_fp32"] is True


def test_script_reports_rmsnorm_fp32_off_when_it_is_off(
    tmp_path: Path, tiny_hf_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tlab.model.layers import RMSNorm

    script = load_script()
    real_load = script.load_qwen3

    def load_with_switch_off(model_dir: Path, device: torch.device) -> nn.Module:
        model = real_load(model_dir, device)
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.compute_fp32 = False
        return model

    monkeypatch.setattr(script, "load_qwen3", load_with_switch_off)
    out = tmp_path / "alignment.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_qwen3.py", "--model-dir", str(tiny_hf_dir), "--out", str(out)],
    )

    script.main()

    # 记的是模型实际的设置，不是写死的值
    assert json.loads(out.read_text(encoding="utf-8"))["rmsnorm_fp32"] is False


def test_src_does_not_depend_on_transformers() -> None:
    # AGENTS 硬性规则 1：transformers 不得出现在 src/tlab 的实现里
    for path in (ROOT / "src" / "tlab").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import transformers" not in text, path
        assert "from transformers" not in text, path
