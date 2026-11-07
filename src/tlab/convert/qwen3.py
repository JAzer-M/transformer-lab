"""把 HuggingFace 上的 Qwen3 权重载入自研模型（M4）。

只读 config.json 和 safetensors 文件，不依赖 transformers（AGENTS 硬性规则 1）。
两边都是 nn.Linear，权重的形状相同，不用转置，只要改名字。
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from tlab.model.gpt import GPT, GPTConfig

# 模型目录要下载的文件
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]

# 每一层里，HF 的名字 → 我们的名字
LAYER_NAMES = {
    "input_layernorm.weight": "ln1.weight",
    "post_attention_layernorm.weight": "ln2.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.out_proj.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}

# 我们的结构只能表示这些取值；不一样时权重也许载得进去，结果却对不上，所以直接报错
REQUIRED = {
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
}


def _rope_theta(hf: Mapping[str, Any]) -> float:
    """RoPE 的底数。旧版配置直接写 rope_theta，新版写在 rope_parameters 里。"""
    if hf.get("rope_scaling") is not None:
        raise ValueError(f"不支持 rope_scaling：{hf['rope_scaling']!r}")
    parameters = hf.get("rope_parameters")
    if parameters is None:
        return float(hf["rope_theta"])
    if parameters.get("rope_type", "default") != "default":
        raise ValueError(f"不支持 rope_parameters：{parameters!r}")
    return float(parameters["rope_theta"])


def gpt_config_from_hf(hf: Mapping[str, Any]) -> GPTConfig:
    """由 HF 的 config.json 得到我们的结构配置。"""
    for key, expected in REQUIRED.items():
        if hf.get(key) != expected:
            raise ValueError(f"{key} 应为 {expected!r}，收到 {hf.get(key)!r}")
    return GPTConfig(
        vocab_size=hf["vocab_size"],
        block_size=hf["max_position_embeddings"],
        d_model=hf["hidden_size"],
        n_layers=hf["num_hidden_layers"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf.get("num_key_value_heads"),
        # 和官方代码一样：配置里没写时是 C / h
        head_dim=hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"],
        qk_norm=True,
        norm="rmsnorm",
        mlp="swiglu",
        mlp_hidden_dim=hf["intermediate_size"],
        pos_encoding="rope",
        rope_theta=_rope_theta(hf),
    )


def convert_state_dict(
    hf_state: Mapping[str, torch.Tensor], n_layers: int
) -> dict[str, torch.Tensor]:
    """把 HF 的名字换成我们的名字。每个 HF 张量都要用上，多一个或少一个都报错。"""
    remaining = dict(hf_state)
    converted: dict[str, torch.Tensor] = {}

    def take(hf_name: str, our_name: str) -> None:
        if hf_name not in remaining:
            raise ValueError(f"权重里缺少 {hf_name}")
        converted[our_name] = remaining.pop(hf_name)

    take("model.embed_tokens.weight", "tok_emb.weight")
    take("model.norm.weight", "ln_f.weight")
    for i in range(n_layers):
        for hf_name, our_name in LAYER_NAMES.items():
            take(f"model.layers.{i}.{hf_name}", f"blocks.{i}.{our_name}")

    # 我们的 lm_head 和词表共用一个参数；文件里单独存了一份时，它必须和词表相同
    lm_head = remaining.pop("lm_head.weight", None)  # (V, C)，文件里可能没有
    if lm_head is not None and not torch.equal(lm_head, converted["tok_emb.weight"]):
        raise ValueError("lm_head.weight 和词表不相同，共用权重的结构表示不了")
    converted["lm_head.weight"] = converted["tok_emb.weight"]

    if remaining:
        raise ValueError(f"权重里有对应不上的张量：{sorted(remaining)[:5]}")
    return converted


def load_qwen3(model_dir: Path, device: torch.device) -> GPT:
    """从一个目录（config.json 和 model.safetensors）建好模型并载入权重，参数是 fp32。"""
    hf = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = gpt_config_from_hf(hf)
    hf_state = load_file(model_dir / "model.safetensors")

    model = GPT(config)
    # load_state_dict 是把数值拷进已有的 fp32 参数，文件里的 bf16 会转成 fp32
    model.load_state_dict(convert_state_dict(hf_state, config.n_layers))
    return model.to(device).eval()
