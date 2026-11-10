# M5 第 1 批：基座评测和对齐

基座是 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，代码是 `d47ab58`。数字都是实际跑出来的，汇总见 `experiments/results.md`。

| 文件 | 在哪跑 | 内容 |
| --- | --- | --- |
| `f1_base_fewshot.json` | RTX 5060，bf16 autocast | 500 题，4 个例子；每题的提示、原始生成文本、抽取的答案和对错 |
| `f1_base_zeroshot.json` | RTX 5060，bf16 autocast | 500 题，不给例子 |
| `base_alignment_cpu_fp32.json` | Mac CPU，fp32 | 和 `transformers` 逐层对比，有门槛 |
| `base_alignment_cuda_fp32.json` | RTX 5060，fp32 | 同上 |
| `base_alignment_cuda_bf16.json` | RTX 5060，bf16 autocast | 只报告差多少，不判定 |
| `rmsnorm_check_cpu_bf16_current.json` | Mac CPU，bf16 autocast | 现在的代码，作为下一行的对照 |
| `rmsnorm_check_cpu_bf16_fp32_inside.json` | Mac CPU，bf16 autocast | **副本实验：** 只把 `RMSNorm` 改成像 HF 一样内部先转 fp32，算完再转回来 |

最后两个文件是在 `git archive` 导出的副本上跑的，副本里没有 `.git`，所以 `git_commit` 记成 `unknown`。代码是 `d47ab58`，第二个文件另外改了 `src/tlab/model/layers.py` 里 `RMSNorm.forward` 的这几行（只在副本里改，仓库里的没动）：

```python
dtype = x.dtype
x = x.float()
ms = x.pow(2).mean(dim=-1, keepdim=True)
x_normed = x * torch.rsqrt(ms + self.eps)
return self.weight * x_normed.to(dtype)
```

这个改动在 CPU 的 bf16 下让每层和分数的误差都变成 0.0；GPU 上没有试过。

## 抽答案规则的试跑（2026-11-07，Mac）

正式评测之前，先在 Mac 上用基座试跑，结果没有单独存档：不给例子的 40 题，旧规则 5/40，新规则 22/40；few-shot 5 题 1/5。旧规则截断续写时还不认 `[Question]` 这种开头，也没有 `answer is` 一条，基座答完接着编题时取到了编出来的数。样本太小，不作结论，没有写进 `experiments/results.md`；这 40 题是看过旧规则的结果后才改的规则。

## 数据来源

评测和训练记录里的题目、标准答案来自 [GSM8K](https://huggingface.co/datasets/openai/gsm8k)（Cobbe et al., 2021），按 MIT 许可证发布，版权归 OpenAI 所有。基座权重 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) 按 Apache-2.0 发布，本仓库不包含权重文件。
