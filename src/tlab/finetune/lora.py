"""LoRA：冻结原来的 Linear， 旁边加一条低秩的旁路（M5）"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        # 原来的 Linear: 权重 W 形状 (out, in), 冻结， 训练时不更新
        self.base = base
        # 整层冻结： 有 bias 时 bias 也不训练（只冻 weight 的话 bias 还会被训练）
        self.base.requires_grad_(False)
        # 记下 r 和 alpha： 存档要用原来的值， 从 alpha / r 反推会有舍入误差
        self.r = r
        self.alpha = alpha
        # 缩放： 换 r 时旁路输出的大小不跟着变
        self.scaling = alpha / r
        # A: (r, in), 和 PEFT 一样用 kaiming_uniform 随机初始化
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: (out, r), 全 0: 刚开始旁路输出是 0， 模型和原来完全一样
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        # 是否已经把旁路加进了 W
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原来的输出
        y = self.base(x)  # (B, T, out)
        # 已经合并时， W 里已经包含旁路， 不能再加一遍
        if self.merged:
            return y
        # 旁路； F.linear(x, A) 就是 x @ A.T, 先降到 r 维
        h = F.linear(x, self.lora_A)  # (B, T, r)
        # 再升回 out 维， 乘缩放后加到原来的输出上
        return y + F.linear(h, self.lora_B) * self.scaling  # (B, T, out)

    @torch.no_grad()
    def merge(self) -> None:
        # 已经合并过就不再加， 否则会加两次
        if self.merged:
            return
        # ΔW = (alpha/r) · B A： (out, r) @ (r, in) = (out, in)， 和 W 形状相同
        self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        # 没合并就不用减
        if not self.merged:
            return
        # 把加进去的 ΔW 减回来
        self.base.weight -= (self.lora_B @ self.lora_A) * self.scaling
        self.merged = False
