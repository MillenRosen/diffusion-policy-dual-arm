# Reproduction Notes

- The original experiments were run in a robosuite-based dual-arm manipulation environment.
- Final evaluation used 50 rollout episodes for the selected Student 2 comparison tables.
- Large datasets and trained checkpoints are not included in this GitHub-ready package.
- A full rerun requires separately supplied demonstration HDF5 files, selected checkpoints, and the original simulator-compatible runtime.
- `src/run_final_dual_arm_eval.py` exposes argparse flags such as `--final-root`, `--episodes`, and `--rollout-max-steps`.
- `src/run_50ep_final_eval_selected.py` preserves the archived script behavior and currently stores `final_root` inside the file rather than through argparse. Update that path before rerunning it outside the original project layout.
