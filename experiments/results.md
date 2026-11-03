| 日期 | 编号 | 配置文件 | commit | 种子 | val loss / 准确率 | tokens/s | 显存峰值 | 一句结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-30 | M1-1 | configs/shakespeare_char.yaml | 54549658efcc8562b56d2505a1f49034eae79a47-dirty | 1337 | val 1.5858 | 105912 | 2597 MB | val 持续下降；样例有剧本格式，词形与语义仍有错误 |
| 2026-11-02 | M2-1 | configs/tinystories_base.yaml | 45e78f1e718f70b322d0e817e47eb8e7272d5773 | 1337 | val 1.3531 | 82756 | 5007 MB | val 持续下降；可生成故事，部分因果矛盾与截断 |
| 2026-11-02 | M3-a0_control-s1337 | configs/ablations/a0_control.yaml（seed=1337） | b7d9989a4642d8a5d702321755fef18399ddd055 | 1337 | val 1.5864 | 82179 | 5007 MB | 对照组；固定 100,007,936 token；未见发散 |
| 2026-11-03 | M3-a0_control-s2024 | configs/ablations/a0_control.yaml（seed=2024） | b7d9989a4642d8a5d702321755fef18399ddd055 | 2024 | val 1.5689 | 80068 | 5007 MB | 对照组；固定 100,007,936 token；未见发散 |
| 2026-11-03 | M3-a2_postln-s1337 | configs/ablations/a2_postln.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.6207（完整验证集 1.6124） | 79285 | 4982 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_postln-s2024 | configs/ablations/a2_postln.yaml（seed=2024） | c72d450997f3e830fe36c895ed6e1bd55448213d | 2024 | val 1.6019（完整验证集 1.6150） | 78602 | 4982 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_rmsnorm-s1337 | configs/ablations/a2_rmsnorm.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.5915（完整验证集 1.5829） | 72475 | 5317 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_rmsnorm-s2024 | configs/ablations/a2_rmsnorm.yaml（seed=2024） | c72d450997f3e830fe36c895ed6e1bd55448213d | 2024 | val 1.5690（完整验证集 1.5814） | 72382 | 5317 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_preln_lr3e-3-s1337 | configs/ablations/a2_preln_lr3e-3.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.4732（完整验证集 1.4651） | 78303 | 5007 MB | 高学习率加测；未发散 |
| 2026-11-03 | M3-a2_postln_lr3e-3-s1337 | configs/ablations/a2_postln_lr3e-3.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 5.8359（完整验证集 5.8343） | 79707 | 4982 MB | 高学习率加测；未发散；验证loss停留在约5.84 |
| 2026-11-03 | M3-a2_rmsnorm_lr3e-3-s1337 | configs/ablations/a2_rmsnorm_lr3e-3.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.4770（完整验证集 1.4694） | 73628 | 5317 MB | 高学习率加测；未发散 |
