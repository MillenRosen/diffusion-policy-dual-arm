# Student 2: Behavior Cloning, DAgger, and Covariate Shift Analysis for Dual-Arm Imitation Learning

## Author

- Name: Yang Muhan
- Student ID: 59793561

## Role Description

This repository packages the Student 2 contribution to the 6019 dual-arm imitation learning project:

- Behavior Cloning baseline
- DAgger baseline
- Noisy-demonstration robustness experiments
- Covariate shift analysis
- Final 50-episode evaluation
- Report figures and appendix diagnostics

## Repository Structure

- `src/`: BC, DAgger, noisy-dataset mixing, final evaluation, expert policy, and diagnostic scripts.
- `results/final_tables/`: lightweight final CSV, JSON, and Markdown summaries, including the 50-episode comparison tables.
- `results/figures/`: final report figures used to communicate the main Student 2 findings.
- `results/appendix/`: compact ablation, noise80 diagnostic, and training-curve support files.
- `report/`: final report PDF, AI Prompt Log, and figure-caption notes retained from the original experiment package.
- `docs/`: packaging notes, reproduction notes, and excluded-file documentation.

## Main Results

At noisy ratio `0.5`:

- BC achieved `0/50` success.
- DAgger achieved `45/50` success.

This result shows that DAgger under moderate noise can reduce covariate shift by using expert-labeled learner-visited states. The packaged tables in `results/final_tables/` preserve the exact summary rows used for the final Student 2 analysis.

## How To Reproduce

The full experiments require the original dual-arm demonstration datasets, checkpoints, and a working robosuite environment. Those large assets are intentionally not included here.

Example commands based on the shipped scripts:

```bash
python src/run_make_mixed_dual_arm.py \
  --clean-hdf5 path/to/dual_arm_clean.hdf5 \
  --noisy-hdf5 path/to/dual_arm_noisy.hdf5 \
  --output-root path/to/mixed_data
```

```bash
python src/run_bc_ratio_experiments.py \
  --mixed-root path/to/mixed_data \
  --output-root path/to/bc_outputs \
  --ratios noise0 noise30 noise50 noise100
```

```bash
python src/run_dagger_ratio_experiments.py \
  --mixed-root path/to/mixed_data \
  --bc-root path/to/bc_outputs \
  --output-root path/to/dagger_outputs \
  --ratios noise0 noise30 noise50 noise100
```

```bash
python src/run_final_dual_arm_eval.py \
  --final-root path/to/final_dual_arm \
  --episodes 20
```

`src/run_50ep_final_eval_selected.py` is also included because it produced the final selected 50-episode comparison artifacts. In the archived project version, it stores `final_root` directly inside the script, so update that path before rerunning it in a new location.

## Notes On Excluded Files

Large demonstration datasets and trained checkpoints are excluded due to file size limits. They can be provided separately if required.

The GitHub package omits HDF5 datasets, model checkpoints, raw intermediate rollout stores, large trace artifacts, temporary logs, and experiment caches. See `docs/excluded_large_files.md` for details.

## AI Usage

The AI Prompt Log is included in `report/Ai Prompt Log Yang Muhan 59793561.docx`.
