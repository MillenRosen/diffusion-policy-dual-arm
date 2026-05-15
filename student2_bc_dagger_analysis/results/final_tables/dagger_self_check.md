# DAgger Self Check

- 是否真的复用了旧版 DAgger 的核心逻辑，而不是偷偷重写成别的东西？
  是。保留了旧版的 `beta mixing`、`student rollout + expert relabel`、`per-round collection -> aggregate -> retrain -> eval` 主循环，只把数据入口、split 和 rollout env 接到 Prompt A 版本。
- 是否和 BC 使用了同一份 mixed HDF5？
  是。mixed HDF5 根目录：`D:\A-6019\project\project_new\experiments\final_dual_arm\mixed_data`。
- 是否和 BC 使用了同一份 train/val split？
  是。DAgger 训练优先读取 mixed manifest 里的 `train_demo_ids / val_demo_ids`；新增 DAgger 数据全部只进入训练集，验证集保持原 mixed val 子集不变。
- eval protocol 是否完全一致？
  是。统一使用 `rollout_episodes=20`、`rollout_max_steps=600`、`rollout_render=False`。
- dual-arm env / expert 是否和当前数据来源一致？
  是。env 读取 mixed HDF5 的 `env/env_info`，expert 复用了旧版 dual-arm scripted expert 控制逻辑，并做了当前 robosuite 版本下的最小兼容修复。
- smoke test 是否通过？ 否。
  smoke test 路径：未找到单独的 smoke metrics 文件。
- noise0 / noise50 / noise100 是否全部正式跑通？ 否。已完成：['noise80']
- 哪一组结果最可疑，需要我人工复核？ `noise80`。
- 还剩下哪些问题没有解决？
  1. 目前只做了 vanilla DAgger，没有做 recovery / stage-aware。
  2. 目前只做了 seed=0，没有多 seed 稳定性。
  3. 当前 best DAgger round 的选择规则是 `success_rate` 优先、`mean_steps` 次优、`best_val_loss` 再次优；如果你后续写论文表格，可能还要再确认最终选轮标准。
  4. DAgger round 的训练清单里会出现重复的 `demo_1/demo_2/...` 名字，这是因为不同 round HDF5 各自从 `demo_1` 重新编号；真正区分它们的是 `source_path`，不影响训练，但汇总展示上还可以再做得更清晰。

## Paths
- BC root: `D:\A-6019\project\project_new\experiments\final_dual_arm\bc_ratio`
- DAgger root: `D:\A-6019\project\project_new\experiments\final_dual_arm\dagger`