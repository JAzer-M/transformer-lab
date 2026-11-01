"""训练配置：dataclass 声明字段，YAML 文件提供取值，命令行可覆盖单个字段。"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


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


LR_SCHEDULES = ("constant", "cosine")


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """读取 YAML；overrides 形如 ["max_iters=5", "device=cpu"]，优先级高于文件。"""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"覆盖项应写成 key=value，收到 {item!r}")
        values[key] = yaml.safe_load(raw)

    # 字段名拼错时直接报错，避免某个超参数被悄悄忽略
    types = {f.name: f.type for f in fields(TrainConfig)}
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"未知配置项：{sorted(unknown)}")

    # 按字段声明的类型转换，例如 YAML 里的 1 → learning_rate 的 1.0
    converted = {key: types[key](value) for key, value in values.items()}
    return TrainConfig(**converted)
