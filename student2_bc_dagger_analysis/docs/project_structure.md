# Project Structure Notes

The original project was scanned from:

`D:\a_6019\report_plus\project_new\project_new`

The clean GitHub submission keeps only lightweight, Student 2-relevant artifacts.

## Source Code Kept

The `src/` directory contains:

- `bc_train.py`: Behavior Cloning training loop, normalization statistics, train/validation split handling, rollout evaluation, and best-checkpoint selection.
- `dagger_train.py`: DAgger data collection, expert relabeling, beta schedule, aggregation loop, retraining loop, and HDF5 bookkeeping fields.
- `mix_hdf5.py`: clean/noisy demonstration mixing at trajectory level.
- `run_make_mixed_dual_arm.py`: batch construction of ratio-specific mixed datasets.
- `run_bc_ratio_experiments.py`: BC experiments across noisy-ratio datasets.
- `run_dagger_ratio_experiments.py`: DAgger experiments across selected noisy-ratio datasets.
- `run_final_dual_arm_eval.py`: summarized final evaluation workflow.
- `run_50ep_final_eval_selected.py`: selected final 50-episode evaluation artifact generation.
- `run_noise0_50ep_sanity_eval.py`: clean-setting sanity evaluation helper.
- `dual_arm_expert.py`: scripted dual-arm expert policy used for DAgger relabeling.
- `diagnose_dagger_questions.py`: targeted DAgger analysis helper.

## Results Kept

The submission preserves:

- Final 50-episode CSV and JSON summaries.
- Main BC vs DAgger comparison tables.
- Best DAgger round summary.
- Sanity-check Markdown and CSV summaries.
- Compact appendix material for BC-only demo-count ablation, noise80 diagnostics, and training-curve notes.

## Figures Kept

The main final report figures retained in `results/figures/` are:

- `bc_vs_dagger_success_50ep_main_no_noise80_v2.png`
- `bc_vs_dagger_mean_steps_50ep_main_no_noise80_v4.png`
- `bc_training_loss_representative_ratios.png`
- `dagger_best_round_summary_no_noise80.png`

## Reports Kept

The `report/` directory includes:

- `6019fa (3).pdf`
- `Ai Prompt Log Yang Muhan 59793561.docx`
- `captions_main_no_noise80.md`

No `.tex` report source was found in the original project scan, so no LaTeX source is included.
