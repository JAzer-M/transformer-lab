| 日期 | 编号 | 配置文件 | commit | 种子 | val loss / 准确率 | tokens/s | 显存峰值 | 一句结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-30 | M1-1 | configs/shakespeare_char.yaml | 54549658efcc8562b56d2505a1f49034eae79a47-dirty | 1337 | val 1.5858 | 105912 | 2597 MB | val 持续下降；样例有剧本格式，词形与语义仍有错误 |
| 2026-11-02 | M2-1 | configs/tinystories_base.yaml | 45e78f1e718f70b322d0e817e47eb8e7272d5773 | 1337 | val 1.3531 | 82756 | 5007 MB | val 持续下降；可生成故事，部分因果矛盾与截断 |
