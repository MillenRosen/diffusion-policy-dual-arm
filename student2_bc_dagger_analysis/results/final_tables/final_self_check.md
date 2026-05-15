# Final Dual-Arm Self Check

1. 使用最终版 dual_arm_clean.hdf5 / dual_arm_noisy.hdf5: Yes
2. 旧实验结果是否删除或归档: 旧 DAgger/final_eval/figures 输出已删除；本轮 DAgger 使用折中设置 dagger_rounds=4, dagger_episodes=20, beta_start=1.0, beta_decay=0.8, beta_min=0.2, rollout_episodes=30，并默认保存失败轨迹。
3. mixed HDF5 是否全部重新生成: Yes, 7 ratios under `D:\A-6019\project\project_new\experiments\final_dual_arm\mixed_data`.
4. 每个 ratio 是否 total_trajectories=160: Yes
5. noise100 是否 160 noisy + 0 clean: Yes
6. manifest 是否包含 step 数和 timestep noisy ratio: Yes
7. BC 是否 7 个 ratio 正式跑通: Yes
8. DAgger 是否跑通 noise0/noise50/noise100: Yes
9. DAgger 是否使用对应 ratio 的 BC checkpoint 初始化: Yes, run_dagger_ratio_experiments used the matching `bc_ratio/<ratio>/seed_0/bc_best.pt`.
10. BC 和 DAgger 是否使用同一份 mixed HDF5 和 train/val split: Yes, both read the same mixed HDF5 manifests; DAgger keeps mixed val split fixed and adds DAgger data only to train.
11. 中间 eval 和 final eval 是否区分清楚: Yes, intermediate CSVs are under `bc_ratio` / `dagger`; 20-episode results are under `final_eval`.
12. 最终报告主表是否只使用 20-episode final evaluation: Yes, use `final_eval/bc_vs_dagger_final_table.csv` and companion final CSVs.
13. 哪一组结果最可疑，需要人工复核: DAgger noise100: round 3 saved 0 episodes, and final success remains low, so this group needs manual review.
14. 当前结果能否直接用于报告: Yes, with the caveat that only seed=0 was run and DAgger noise100 is weak/suspicious.

Note: `demonstrations_change` still contains single-arm and sequential raw HDF5 files, but this run only references the dual-arm clean/noisy files. Non-final old project HDF5 outputs were deleted.