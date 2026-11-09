# 面试问答准备

`docs/PLAN.md` 里列的 8 道高频题。**答案由你自己用自己的话写**，写在每题的“你的回答”下面；这里只放素材：对应的代码、实测的数字和相关的自测题（`docs/self-test-qa.md` 里有参考答案）。数字来自 `experiments/` 下的结果文件和 [实验报告](report.md)；手算或另外统计的数字注明了出处。

## 1. 现场手写 multi-head attention

- **代码：** `src/tlab/model/attention.py`（你亲手写的），测试 `tests/test_attention.py` 里有和 `F.scaled_dot_product_attention` 的数值对照和因果性测试。
- **要能讲清的形状：** (B, T, C) → 投影 → 拆成 (B, h, T, d) → 分数 (B, h, T, T) → 加因果掩码 → softmax → 乘 v → 合回 (B, T, C) → `out_proj`。
- **相关自测题：** 第 19–21 题（每一步的形状、除以 √d、因果掩码）、第 42 题（GQA 时 k 的每一步形状）。

你的回答：

## 2. KV cache 的原理和显存；GQA 为什么能省显存

- **代码：** `src/tlab/model/attention.py` 的 `KVCache` 和 RoPE 的 `offset`，`src/tlab/eval/generate.py` 的 `use_cache`。
- **数字：** 6 层、T=512、d=64、bf16 时完整缓存：MHA（g=6）4.50 MiB、GQA-2 1.50 MiB（实测和手算一致），MQA（g=1）手算 0.75 MiB。开缓存的提速（对照结构）：RTX 5060 上 B=1、8、32 是 1.00、1.85、6.80 倍，CPU 上 B=1 是 7.42 倍；B=32 时开缓存让显存峰值从 662 MiB 降到 267 MiB。GQA 本身的收益：B=32、有缓存时峰值从对照的 267 MiB 降到 167 MiB。
- **相关自测题：** 第 44–48 题。

你的回答：

## 3. RoPE 和绝对位置编码的区别

- **代码：** `src/tlab/model/layers.py` 的 `RotaryEmbedding`（你写的）。
- **数字：** 同 1 亿 token，RoPE 比可学习位置表的 val loss 低 0.0523，参数少 196,608（没有位置表）。2 倍训练长度上，RoPE 后半段 loss 2.30 / 2.13，正弦 4.93 / 5.06。
- **相关自测题：** 第 37–41 题（第 38 题的三行推导最值得会）。

你的回答：

## 4. Pre-Norm 和 Post-Norm 的区别，实验看到了什么

- **代码：** `src/tlab/model/block.py`（你写的）的 `norm_position` 开关。
- **数字：** 学习率 6e-4 时 Post-LN 比 Pre-LN 的 val loss 高 0.0338；学习率 3e-3 时 Post-LN 停在 5.83–5.85，接近只按词频猜的水平，Pre-LN 正常收敛。
- **相关自测题：** 第 33 题（含“为什么 Post-LN 不要 `ln_f`”和“学不动”是什么意思）。

你的回答：

## 5. LoRA 的原理、初始化、rank 怎么选

- **代码：** `src/tlab/finetune/lora.py` 的 `LoRALinear`（你写的），`src/tlab/finetune/lora_model.py`。
- **数字：** 每层 q、k、v、out 共 10,240·r 个参数，28 层 286,720·r；r=4 到 32 占全部参数的 0.19%–1.54%。F2：r=4、8、16、32 的准确率均值 49.0%、49.2%、50.5%、50.3%，差异不显著。F4：和 PEFT 训练 454 步逐位相同。
- **相关自测题：** 第 51–56 题。

你的回答：

## 6. 预训练和 SFT 的 loss 计算有什么不同

- **代码：** `src/tlab/finetune/sft_labels.py` 的 `build_example`（你写的），`src/tlab/finetune/sft.py` 的 `sft_step`。
- **数字：** GSM8K 训练集的提示平均 63.4 个 token、答案 91.2 个（M5 时用 Qwen3 的分词器统计，见第 57 题参考答案）；只有答案和最后的 eos 算 loss。梯度累积按整步的答案 token 数加权，2 × 8 和 4 × 4 第 1 步的 loss 都是 0.92283。
- **相关自测题：** 第 57、58、60 题。

你的回答：

## 7. 权重对齐时最难排查的 bug 是什么，怎么定位的

- **代码：** `src/tlab/convert/qwen3.py`，`scripts/check_qwen3.py`（逐层对比）。
- **素材：** `head_dim` 不等于 C/h；QK-Norm 必须在旋转之前，而 `weight` 全是 1 时顺序看不出来；`lm_head` 单独存了一份；形状相同的权重写反只能靠逐层对比发现；fp32 下最后剩下的 1.73e-04 来自除以 √d 的写法；bf16 下 0.65625 的差来自 RMSNorm 没转 fp32。
- **相关自测题：** 附 1、附 2、第 67 题。

你的回答：

## 8. 实验的局限；如果有更多算力下一步做什么

- **素材：** [实验报告](report.md) 的第 7、8 节：规模小、每组 2 个种子、500 题的标准误约 2.2 个百分点、F3 和 F5 没做、评测一次只答一题。SFT 后准确率提升在噪声范围内，这一点要能讲清楚原因（第 65 题）。

你的回答：
