这部分主要作为经典模仿学习 baseline 和中间对照方法，为后续 的 Diffusion Policy 提供比较基准和问题动机。

## 文件夹内容

- `code/`：BC、DAgger、scripted/noisy expert 数据采集与补充评估脚本。
- `figures/`：训练 loss 曲线、BC vs DAgger 成功率对比图，以及 covariate shift 分析图。
- `metrics/`：BC、DAgger、稳定评估和 covariate shift 分析的 JSON/CSV 指标文件。
- `models/`：用于对比的关键模型 checkpoint。
- `dagger_data/`：noisy-data DAgger 实验中收集到的聚合 HDF5 数据。

## 主要实验结果

- noisy demonstrations 上的 BC baseline 成功率为 `0.20`。
- 最好的 DAgger 轮次是 `DAgger R2`，成功率为 `0.40`。
- 最好的 DAgger checkpoint 保存在 `models/dagger_noisy_round_2_best.pt`。

这说明在当前实验设置下，DAgger best round 相比 noisy BC baseline 有一定提升。不过这个提升主要应作为 baseline 对照和趋势性结果使用。

## 10-episode 稳定评估补充

为了避免只依赖 5-episode 训练阶段评估，额外对 noisy BC 和 DAgger round 2 checkpoint 做了 10-episode 评估：

- BC noisy：`0.00`，即 0/10。
- DAgger round 2：`0.10`，即 1/10。

这个更保守的评估仍然显示 DAgger round 2 略好于 noisy BC，但差距较小。因此报告中建议表述为：DAgger best round 展示了弱提升趋势，但结果仍然受到评估方差和聚合轮次的影响。

## 结果解释建议

这部分的定位是经典 imitation learning baseline，而不是项目最终性能最高的模型。BC 提供最基础的监督学习 baseline；DAgger 通过收集 student policy 访问到的状态，并用 scripted expert 重新标注动作，作为缓解 covariate shift 的中间方法。

在 noisy-data 设置下，BC 成功率为 `20%`，DAgger round 2 成功率为 `40%`，但 DAgger round 3 出现回落。因此得出以下结论：

- best-round DAgger 相比 BC 有一定改善。
- DAgger 对 beta schedule、数据聚合轮次和评估随机性比较敏感。
- 这也自然引出 3 号 Diffusion Policy 的必要性：更强的序列建模和多模态动作建模有望提升长期任务稳定性。

## Covariate Shift 分析

covariate shift 分析比较了 noisy BC 数据和 DAgger 收集状态相对于 scripted expert 状态分布的差异：

- `figures/covariate_shift_state_pca.png`：状态分布的 PCA 投影图。
- `figures/covariate_shift_nearest_distance.png`：到 scripted expert 状态分布的最近距离统计图。
- `metrics/covariate_shift_metrics.json`：对应的数值统计结果。

距离统计如下：

```json
{
  "noisy BC data": {
    "mean": 0.8968102931976318,
    "median": 0.6526094675064087,
    "p90": 1.9717308759689334
  },
  "DAgger R1": {
    "mean": 0.3654256761074066,
    "median": 0.32169100642204285,
    "p90": 0.6247041702270508
  },
  "DAgger R2": {
    "mean": 1.5482572317123413,
    "median": 0.3282237648963928,
    "p90": 6.247471094131473
  },
  "DAgger R3": {
    "mean": 3.747159719467163,
    "median": 0.55782151222229,
    "p90": 14.339627742767341
  }
}
```

其中 R1 的状态分布更接近 scripted expert，R2/R3 的 mean 和 p90 变大，说明后续轮次中出现了更多偏离 expert distribution 的状态。这可以用于说明 DAgger 确实暴露了 student policy 访问到的状态分布，但后续轮次也可能引入更大的分布漂移。

做 Diffusion Policy 时，建议主要使用以下结果作为对比：

- BC noisy baseline：`metrics/bc_noisy_metrics.json`。
- DAgger best round：`metrics/dagger_noisy_metrics.json` 中的 DAgger R2。
- 对比表：`metrics/bc_dagger_noisy_comparison.csv`。
- 成功率对比图：`figures/success_rate_bc_vs_dagger_noisy.png`。
- 稳定评估结果：`metrics/stable_eval_noisy_bc_vs_dagger_round2.json`。
- covariate shift 分析图：`figures/covariate_shift_state_pca.png` 和 `figures/covariate_shift_nearest_distance.png`。

