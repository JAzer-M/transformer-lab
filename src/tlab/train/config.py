"""训练配置：dataclass 声明字段，YAML 文件提供取值，命令行可覆盖单个字段。"""

from dataclasses import dataclass, fields
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, TypeVar, get_args

import yaml

from tlab.model.gpt import check_rmsnorm_fp32
from tlab.model.layers import (
    check_block_settings,
    check_kv_heads,
    check_pos_encoding,
    check_qwen3_switches,
)


@dataclass
class TrainConfig:
    # 数据与输出
    data_dir: str = "data/shakespeare_char"
    out_dir: str = "runs/shakespeare_char"
    # 模型结构（vocab_size 从数据目录的 meta.json 读取）
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    block_size: int = 256
    # 优化
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_iters: int = 2000
    # M2 新增；默认值与 M1 行为相同，旧配置文件不必改
    grad_accum_steps: int = 1  # 每次更新前累积几个 micro batch
    grad_clip: float = 0.0  # 梯度总范数上限；0 表示不裁剪
    lr_schedule: str = "constant"  # "constant" 或 "cosine"
    warmup_iters: int = 0  # 前多少步把学习率从小线性升到 learning_rate
    min_lr: float = 0.0  # cosine 在最后一步（第 max_iters - 1 步）降到的学习率
    # M3 新增（消融开关）；默认值与 M1/M2 结构相同
    norm: str = "layernorm"  # "layernorm" 或 "rmsnorm"
    norm_position: str = "pre"  # "pre"（子层前）或 "post"（加法后，不用 ln_f）
    mlp: str = "gelu"  # "gelu"（M1/M2 的 MLP）或 "swiglu"
    pos_encoding: str = "learned"  # "learned"、"sinusoidal" 或 "rope"
    rope_theta: float = 10000.0  # 只在 pos_encoding 为 "rope" 时有用
    n_kv_heads: int | None = None  # k、v 的组数；None 表示等于 n_heads（MHA）
    head_dim: int | None = None  # 每个头的维度；None 表示 d_model // n_heads
    qk_norm: bool = False  # q、k 拆头后各做一次 RMSNorm（Qwen3 的做法）
    mlp_hidden_dim: int | None = None  # SwiGLU 的隐藏维度；None 表示 8C//3
    # RMSNorm 内部先转 fp32（Qwen3 的做法）；模型的开关都要在这里有同名字段，
    # 存档和续训的结构检查靠它。TinyStories 的配置都不写，默认关闭
    rmsnorm_fp32: bool = False
    # 存档：0 表示只在训练结束时存；resume=True 时从 out_dir/ckpt.pt 接着训练
    ckpt_interval: int = 0
    resume: bool = False
    # 评估
    eval_interval: int = 250
    eval_iters: int = 50
    # 运行环境；"auto" 表示按 cuda → mps → cpu 选择
    seed: int = 1337
    device: str = "auto"

    def __post_init__(self) -> None:
        # 调度名拼错时在建配置这一刻就报错，不用等到训练开始
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule 应为 {LR_SCHEDULES} 之一，收到 {self.lr_schedule!r}"
            )
        check_block_settings(self.norm, self.norm_position, self.mlp)
        check_pos_encoding(self.pos_encoding, self.rope_theta, self.head_dim)
        check_kv_heads(self.n_heads, self.n_kv_heads)
        check_qwen3_switches(self.mlp, self.head_dim, self.qk_norm, self.mlp_hidden_dim)
        check_rmsnorm_fp32(self.rmsnorm_fp32, self.norm, self.qk_norm)


LR_SCHEDULES = ("constant", "cosine")


ConfigT = TypeVar("ConfigT")


def _convert(declared: Any, value: Any) -> Any:
    """按字段声明的类型转换，例如 YAML 里的 1 → learning_rate 的 1.0。

    int | None 这种可选字段不能直接调用：None 保持 None，其他值按 int 转。
    整数字段不接受 2.9 和 true：int() 会悄悄变成 2 和 1，等于悄悄换了结构。
    布尔字段只接受 true 和 false：bool() 只要不是空值都得到 True，拼错的取值会被当成开启。
    """
    if isinstance(declared, UnionType):
        if value is None:
            return None
        declared = next(t for t in get_args(declared) if t is not NoneType)
    if declared is int and (
        isinstance(value, bool) or (isinstance(value, float) and not value.is_integer())
    ):
        raise ValueError(f"应为整数，收到 {value!r}")
    if declared is bool and not isinstance(value, bool):
        raise ValueError(f"应为布尔值 true 或 false，收到 {value!r}")
    return declared(value)


def load_dataclass(
    config_class: type[ConfigT], path: str | Path, overrides: list[str] | None = None
) -> ConfigT:
    """读取 YAML 填进 config_class；overrides 形如 ["max_iters=5", "device=cpu"]，优先级高于文件。"""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"覆盖项应写成 key=value，收到 {item!r}")
        values[key] = yaml.safe_load(raw)

    # 字段名拼错时直接报错，避免某个超参数被悄悄忽略
    types = {f.name: f.type for f in fields(config_class)}
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"未知配置项：{sorted(unknown)}")

    converted = {key: _convert(types[key], value) for key, value in values.items()}
    return config_class(**converted)


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """读取训练配置。"""
    return load_dataclass(TrainConfig, path, overrides)
