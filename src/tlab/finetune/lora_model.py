"""把 LoRA 装进模型、只存 A 和 B、合并回普通模型（M5，agent 编写）。

LoRALinear 本身由用户手写在 lora.py；这里只管“装在哪里、怎么存、怎么合并”。
存档只有 A、B 和三个设置，不含基座。载入时先载基座，再调用 load_lora。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tlab.convert.qwen3 import load_qwen3
from tlab.finetune.lora import LoRALinear
from tlab.model.gpt import GPT
from tlab.train.utils import file_sha256

# 能装 LoRA 的层：注意力里的四个投影；F3 要试前馈子层时再加
SUPPORTED_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
# 不指定时四个都装（计划里的默认）
DEFAULT_TARGETS = SUPPORTED_TARGETS


def _lora_children(model: nn.Module) -> list[tuple[nn.Module, str, LoRALinear]]:
    """找出所有 LoRALinear，连同它的父模块和属性名，合并时要把它换回去。"""
    return [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, LoRALinear)
    ]


def _find_targets(
    model: nn.Module, targets: Sequence[str]
) -> list[tuple[str, nn.Module, str]]:
    """找出要换的层：(父模块名, 父模块, 属性名)，按模型里的顺序；不改模型。"""
    if not targets:
        raise ValueError("targets 不能为空")
    unknown = sorted(set(targets) - set(SUPPORTED_TARGETS))
    if unknown:
        raise ValueError(f"不认识的 targets：{unknown}，只能是 {SUPPORTED_TARGETS}")
    if _lora_children(model):
        # 再装一次会把 LoRALinear 里的 base 也包一层
        raise ValueError("模型里已经装过 LoRA")
    found = [
        (module_name, module, target)
        for module_name, module in model.named_modules()
        for target in SUPPORTED_TARGETS
        if target in targets and isinstance(getattr(module, target, None), nn.Linear)
    ]
    if not found:
        raise ValueError(f"模型里没有可以替换的层：{list(targets)}")
    return found


def _full_name(parent_name: str, attribute: str) -> str:
    return f"{parent_name}.{attribute}" if parent_name else attribute


def apply_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> list[str]:
    """冻结整个模型，把名字在 targets 里的 Linear 换成 LoRALinear；返回替换了哪些层。

    先找齐要换的层再动模型，出错时模型保持原样。
    """
    found = _find_targets(model, targets)
    # 词表、归一化、前馈子层和 W 全部冻结；LoRALinear 新建的 A、B 默认要训练
    model.requires_grad_(False)
    for _, module, target in found:
        base = getattr(module, target)
        # A、B 建在 CPU 上（随机数和设备无关），再搬到 W 所在的设备；类型保持 fp32
        lora = LoRALinear(base, r, alpha).to(device=base.weight.device)
        setattr(module, target, lora)
    return [_full_name(name, target) for name, _, target in found]


def count_trainable(model: nn.Module) -> int:
    """要训练的参数个数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_settings(model: nn.Module) -> dict[str, Any]:
    """从模型里读出 r、alpha 和装在哪些层：每个 LoRALinear 必须相同。"""
    children = _lora_children(model)
    if not children:
        raise ValueError("模型里没有 LoRA")
    ranks = {child.r for _, _, child in children}
    alphas = {child.alpha for _, _, child in children}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError(f"各层的 r 或 alpha 不同：r {ranks}，alpha {alphas}")
    (r,) = ranks
    (alpha,) = alphas
    names = {name for _, name, _ in children}
    return {
        "r": r,
        "alpha": alpha,
        "targets": [t for t in DEFAULT_TARGETS if t in names],
    }


def lora_weights(model: nn.Module) -> dict[str, nn.Parameter]:
    """模型里所有的 A、B，键是参数名，比如 blocks.0.attn.q_proj.lora_A。"""
    weights: dict[str, nn.Parameter] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            weights[f"{name}.lora_A"] = module.lora_A
            weights[f"{name}.lora_B"] = module.lora_B
    return weights


def save_lora(model: nn.Module, path: Path) -> None:
    """只存 A、B 和三个设置，不存基座。"""
    weights = {
        name: parameter.detach().cpu().clone()
        for name, parameter in lora_weights(model).items()
    }
    torch.save({**lora_settings(model), "weights": weights}, path)


def load_lora(model: nn.Module, path: Path) -> dict[str, Any]:
    """给已经载好基座的模型装上 LoRA，再载入存档里的 A、B；返回存档里的设置。"""
    saved = torch.load(path, weights_only=True, map_location="cpu")
    settings = {key: saved[key] for key in ("r", "alpha", "targets")}
    r = settings["r"]

    # 先全部检查完再动模型：出错时模型保持原样
    expected: dict[str, tuple[int, int]] = {}
    for parent_name, module, target in _find_targets(model, settings["targets"]):
        base = getattr(module, target)
        name = _full_name(parent_name, target)
        expected[f"{name}.lora_A"] = (r, base.in_features)  # (r, in)
        expected[f"{name}.lora_B"] = (base.out_features, r)  # (out, r)
    missing = sorted(set(expected) - set(saved["weights"]))
    unexpected = sorted(set(saved["weights"]) - set(expected))
    if missing or unexpected:
        # 不能悄悄留着随机初始化的 A
        raise ValueError(f"存档和模型对不上：模型多出 {missing}，存档多出 {unexpected}")
    for name, shape in expected.items():
        if tuple(saved["weights"][name].shape) != shape:
            raise ValueError(
                f"{name} 的形状是 {tuple(saved['weights'][name].shape)}，"
                f"按 r={r} 应为 {shape}"
            )

    apply_lora(model, r, settings["alpha"], settings["targets"])
    with torch.no_grad():
        for name, parameter in lora_weights(model).items():
            parameter.copy_(saved["weights"][name])
    return settings


def merge_lora(model: nn.Module) -> nn.Module:
    """把每个旁路加进 W，再换回原来的 Linear：参数名和没装 LoRA 时完全相同。

    W 必须是 fp32：bf16 只有约 3 位有效数字，ΔW 比 W 小得多，加进去会被舍掉一部分。
    """
    children = _lora_children(model)
    if not children:
        raise ValueError("模型里没有 LoRA")
    low = sorted({str(c.base.weight.dtype) for _, _, c in children} - {"torch.float32"})
    if low:
        raise ValueError(f"合并要在 fp32 下做，W 现在是 {low}")
    for parent, name, child in children:
        child.merge()
        setattr(parent, name, child.base)
    return model


def check_lora_file(lora: str | Path | None) -> None:
    """LoRA 存档不存在就报错。脚本在找或下载基座之前先调用：路径写错应该马上知道。"""
    if lora is not None and not Path(lora).is_file():
        raise FileNotFoundError(f"找不到 LoRA 存档：{lora}")


def load_qwen3_with_lora(
    model_dir: Path, device: torch.device, lora: str | Path | None
) -> tuple[GPT, dict[str, Any] | None]:
    """载入 Qwen3 基座；给了 LoRA 存档就装上并合并。返回模型和存档的记录（没有就是 None）。

    合并后和普通模型一样快；W 是 fp32，合并不会丢精度。
    """
    check_lora_file(lora)
    model = load_qwen3(model_dir, device)
    if lora is None:
        return model, None
    # 记下哈希：同一个路径的存档被重新训练覆盖后，也分得清用的是哪一份
    info = {"path": str(lora), "sha256": file_sha256(Path(lora))}
    info |= load_lora(model, Path(lora))
    merge_lora(model)
    return model, info
