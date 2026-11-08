"""F4：用 HF + PEFT 跑同样的 LoRA SFT，和自研 LoRA 对比 loss 曲线。

两边只有 LoRA 的实现不同，其余全部相同：同一份基座权重、同一个起点（init_lora，
自研那次训练存下的 lora_init.pt）、同样的数据和顺序（prepare_items、training_plan）、
同一个训练循环和 loss（sft.fit、sft_step：只算答案，按 token 加权）。
不用 HF 的 labels= 和 DataCollator：pad 和 eos 是同一个编号，最后那个 eos 会被遮掉。
peft 和 transformers 只在这个脚本里用，src/tlab 不依赖它们（AGENTS 硬性规则 1）。

用法：
    uv run python scripts/train_peft.py --config configs/finetune/f4_peft_r8.yaml
输出在 out_dir：metrics.csv、run.json、lora_init.pt、lora.pt。存档用我们的参数名，
可以直接装进自研模型，用 eval_gsm8k.py 评测。

环境里设了 SOCKS 代理（ALL_PROXY=socks5://...）时，下载会报缺 socksio。如果同时有 HTTP
代理，在命令前加 env -u ALL_PROXY -u all_proxy，这条命令就只走 HTTP 代理。
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Qwen3ForCausalLM

from tlab.finetune.sft import (
    SFTConfig,
    check_start,
    fit,
    load_sft_config,
    prepare_items,
    resolve_device,
    resolve_model_dir,
)
from tlab.finetune.sft_data import Item
from tlab.tokenizer.hf_bpe import HFTokenizer

# 我们的投影名 → HF 的投影名：只有 out_proj 叫法不同
HF_PROJECTIONS = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "out_proj": "o_proj",
}
OUR_NAME = re.compile(r"blocks\.(\d+)\.attn\.(\w+)\.(lora_A|lora_B)")


def peft_name(our_name: str) -> str:
    """blocks.1.attn.out_proj.lora_B → base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weight"""
    match = OUR_NAME.fullmatch(our_name)
    if match is None or match.group(2) not in HF_PROJECTIONS:
        raise ValueError(f"不认识的 LoRA 参数名字：{our_name!r}")
    layer, projection, which = match.groups()
    return (
        f"base_model.model.model.layers.{layer}.self_attn."
        f"{HF_PROJECTIONS[projection]}.{which}.default.weight"
    )


class LogitsOnly(nn.Module):
    """把 HF 模型包一层：输入 (B, T) 的 token，只返回 (B, T, V) 的分数，和自研模型一样。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=idx, use_cache=False).logits  # (B, T, V)


def build_peft_model(
    model_dir: Path, config: SFTConfig, device: torch.device
) -> nn.Module:
    """载入官方实现的 Qwen3（fp32、eager 注意力，和 M4 对齐时一样），用 PEFT 装上 LoRA。"""
    base = Qwen3ForCausalLM.from_pretrained(
        model_dir, attn_implementation="eager", dtype=torch.float32
    ).to(device)
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        target_modules=[HF_PROJECTIONS[t] for t in config.targets],
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(base, lora_config)


def _peft_lora_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if ".lora_" in n}


def copy_into_peft(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """把我们名字下的 A、B 拷进 PEFT 模型；多一个、少一个或形状不同都报错。

    两边的形状本来就一样：A 是 (r, in)，B 是 (out, r)，不用转置。
    """
    targets = _peft_lora_parameters(model)
    mapped = {peft_name(name): (name, tensor) for name, tensor in weights.items()}
    missing = sorted(set(targets) - set(mapped))
    unexpected = sorted(set(mapped) - set(targets))
    if missing or unexpected:
        raise ValueError(f"A、B 对不上：PEFT 多出 {missing}，存档多出 {unexpected}")
    with torch.no_grad():
        for target, (name, tensor) in mapped.items():
            if targets[target].shape != tensor.shape:
                raise ValueError(f"{name} 的形状 {tuple(tensor.shape)} 和 PEFT 的不同")
            targets[target].copy_(tensor)  # A 是 (r, in)，B 是 (out, r)


def export_from_peft(model: nn.Module) -> dict[str, torch.Tensor]:
    """PEFT 里的 A、B，换回我们的名字，可以直接用 load_lora 装进自研模型。"""
    pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)\.(lora_A|lora_B)\.default\.weight"
    )
    ours = {hf: our for our, hf in HF_PROJECTIONS.items()}
    exported = {}
    for name, parameter in _peft_lora_parameters(model).items():
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"不认识的 PEFT 参数名字：{name!r}")
        layer, projection, which = match.groups()
        key = f"blocks.{layer}.attn.{ours[projection]}.{which}"
        exported[key] = (
            parameter.detach().cpu().clone()
        )  # 形状不变：(r, in) 或 (out, r)
    return exported


def train_peft(
    model: nn.Module,
    train_items: list[Item],
    val_items: list[Item],
    config: SFTConfig,
    device: torch.device,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    """用和自研 LoRA 完全相同的训练循环训练 PEFT 模型；存档换成我们的名字。"""
    settings = {"r": config.r, "alpha": config.alpha, "targets": list(config.targets)}

    def save(path: Path) -> None:
        torch.save({**settings, "weights": export_from_peft(model)}, path)

    return fit(
        LogitsOnly(model),
        train_items,
        val_items,
        config,
        device,
        save_weights=save,
        extra_info={
            "implementation": "peft",
            "peft_version": peft.__version__,
            "transformers_version": transformers.__version__,
            "lora": settings,
            **extra_info,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--model-dir", type=Path, default=None, help="不给就下载")
    parser.add_argument("overrides", nargs="*", help="覆盖单个字段，如 max_steps=5")
    args = parser.parse_args()

    config = load_sft_config(args.config, args.overrides)
    if config.init_lora is None:
        # 没有共同的起点，两条曲线本来就不会重合
        sys.exit("需要 init_lora：自研那次训练存下的 lora_init.pt")
    if config.base_dtype != "float32":
        sys.exit("PEFT 对照只支持 base_dtype=float32，和自研那次训练一致")
    device = resolve_device(config)
    model_dir, model_name = resolve_model_dir(config.repo, args.model_dir)

    start = torch.load(Path(config.init_lora), weights_only=True, map_location="cpu")
    init = check_start({k: start[k] for k in ("r", "alpha", "targets")}, config)

    tokenizer = HFTokenizer.load(model_dir / "tokenizer.json")
    train_items, val_items, data_info = prepare_items(config, tokenizer)
    model = build_peft_model(model_dir, config, device)
    copy_into_peft(model, start["weights"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    info = train_peft(
        model,
        train_items,
        val_items,
        config,
        device,
        extra_info={
            "model": model_name,
            "init_lora": init,
            **data_info,
        },
    )
    print(
        f"完成 {info['steps_run']}/{info['total_steps']} 步：train {info['final_train_loss']:.4f}，"
        f"val {info['final_val_loss']:.4f}，{info['tokens_per_second']:.0f} tokens/s，"
        f"显存峰值 {info['peak_memory_mb']} MiB"
    )


if __name__ == "__main__":
    main()
