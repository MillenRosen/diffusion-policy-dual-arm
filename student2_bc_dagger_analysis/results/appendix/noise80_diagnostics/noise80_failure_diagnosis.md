# noise80 DAgger 0/50 failure diagnosis

## Scope
All diagnostics were written under this folder only:
`D:\A-6019\project\project_new\experiments\final_dual_arm\80_check`

No original training code, checkpoints, mixed HDF5 files, or existing experiment results were modified.

## Short conclusion
The immediate, verified reason noise80 DAgger gets 0/50 is not a bad manifest, wrong checkpoint path, stale result, or wrong DAgger label. The learned noise80 policy approaches the handles but fails to transition into a stable grasp-and-lift phase: during rollout its gripper actions remain negative/open and its vertical lift commands stay near zero or negative. Consequently the pot is never lifted enough to satisfy `env._check_success()`.

This was confirmed by a counterfactual rollout: using the same noise80 round-3 policy and same initial states, but forcing gripper close once both arms are near the handles and then forcing upward z lift, succeeded in 2/3 episodes. Therefore the policy often reaches a position where success is physically recoverable, but the learned controller does not issue the required close/lift actions.

## File/path sanity checks
Source: `static_audit_summary.txt`, `static_audit.json`

- noise80 mixed manifest is correct: 32 clean + 128 noisy = 160 trajectories.
- trajectory-level noisy ratio is 0.8.
- timestep-level noisy ratio is 0.845957785742732.
- train split: 28 clean / 100 noisy trajectories, 3391 clean steps / 16391 noisy steps.
- val split: 4 clean / 28 noisy trajectories, 477 clean steps / 4851 noisy steps.
- DAgger used final paths under `experiments/final_dual_arm`.
- DAgger ran 4 rounds for noise80, no skipped rounds.
- noise80 intermediate policy-only eval was 0/30 for all four rounds.
- best round = 1 is a tie-break result because every round had intermediate success rate 0; round1 had the lowest validation loss.

## DAgger aggregate and label checks
Source: previous HDF5 attr checks plus `static_audit_summary.txt`

- Each noise80 DAgger round saved 20 episodes.
- Round collection success counts: 20, 20, 20, 19.
- Collection success is beta-mixed / expert-assisted, not student-only performance.
- `aggregate_data_paths` include the initial mixed HDF5 plus all previous/current DAgger round HDF5s for each round.
- HDF5 `actions_are = expert_actions` and `action_label_dataset = actions`.
- First-demo checks showed `actions == expert_actions`.

## Closed-loop rollout evidence
Source: `rollout_trace_summary_1ep.csv`, `trace_3ep_comparison.csv`, `expert_stage_diagnostics_3ep.csv`

### Successful reference: noise50 round3
Across 3 diagnostic episodes, noise50 round3 succeeded every time:

- first policy close steps: 121, 130, 139
- first policy lift steps: 132, 141, 151
- max gripper values: 1.0 / 1.0
- max z actions: about 0.82-0.85 on both arms
- pot z increase: about 0.093-0.099

### Failing case: noise80 round3
Across 3 diagnostic episodes, noise80 round3 failed every time while still reaching the handles:

- min both-arm handle distance: 0.0202, 0.0109, 0.0218
- first distance < 0.025 occurred at steps: 133, 135, 103
- first policy close: never
- first policy lift: never
- max gripper values stayed negative:
  - ep0: -0.3775 / -0.3770
  - ep1: -0.4342 / -0.4406
  - ep2: -0.5064 / -0.5099
- max z actions stayed tiny:
  - ep0: 0.0201 / 0.0194
  - ep1: 0.0389 / 0.0281
  - ep2: 0.0070 / 0.0261
- pot z increase was effectively zero:
  - ep0: 0.00021
  - ep1: 0.0
  - ep2: 0.00040

### Partial reference: noise100 round3
noise100 round3 failed in episodes 0-1 but succeeded in episode 2:

- successful ep2 first policy close: step 197
- successful ep2 first policy lift: step 222
- max gripper values: 1.0 / 1.0
- max z actions: 0.880 / 0.872
- pot z increase: 0.0938

This shows the difference between failure and success is the emergence of the close/lift transition, not merely proximity to the handles.

## Counterfactual intervention
Source: `forced_grasp_lift_noise80_r3_3ep.csv`

Experiment: execute the same noise80 round3 policy, but once both arms are within 0.025 of the handles, force both grippers closed; after 20 additional steps, force both z actions to 0.8 while keeping the rest of the policy action.

Results:

| episode | success | trigger_step | lift_step | max_pot_z_delta |
|---:|:---:|---:|---:|---:|
| 0 | True | 133 | 153 | 0.1075 |
| 1 | False | 135 | 155 | 0.0019 |
| 2 | True | 103 | 123 | 0.1026 |

Interpretation: the environment and success check are capable of registering success from the noise80 policy's visited region when the missing grasp/lift commands are supplied. Thus the primary failure mode is the learned policy's failure to trigger grasp/lift, not a broken evaluator or impossible initial states.

## Why noise80 is worse than noise100
The result is non-monotonic, but the diagnostics support this explanation:

- noise80 is a mixed distribution, not a clean interpolation between noise50 and noise100.
- The noise80 train/val split is dominated by noisy trajectories and noisy timesteps, with very little clean signal: only 28 clean train trajectories and 4 clean validation trajectories.
- DAgger adds 20 episodes per round, but the aggregate remains dominated by the noisy base data. Validation loss worsens monotonically across DAgger rounds: 0.7479, 0.8265, 0.9011, 0.9871.
- Offline action error on sampled aggregate states is not catastrophic, which means the model can imitate labels on familiar states. The failure appears under closed-loop rollout: small errors move the system into pre-grasp states where the learned policy keeps the grippers open and never lifts.
- noise100 is all-noisy and therefore more distributionally coherent; its round3 occasionally enters the close/lift mode, giving 12/50 final successes. noise80 combines mostly noisy behavior with a small clean subset, which likely creates a conflicting / averaged phase-switch policy that approaches but does not commit to grasp/lift.

## Answer to the original diagnostic questions
A. noise80 mixed data is correct: yes.

B. noise80 DAgger really ran 4 rounds: yes, no skipped rounds.

C. Are all rounds 0/50? Round1 is confirmed 0/50. Rounds 2-4 have policy-only intermediate eval 0/30 each; full 50ep for rounds 2-4 was not rerun because the smaller diagnostics already identify the failure mechanism and avoid long runs. There is no evidence that later rounds are better; their validation losses are worse and intermediate eval is 0/30.

D. Did final summary choose the wrong round? No. Since all four intermediate success rates are 0, the selection rule falls through to lower mean steps and lower validation loss; all mean steps are 600, so round1 wins by best validation loss.

E. Is there old result contamination? No evidence. Paths in metrics and manifests point to `experiments/final_dual_arm`, and the HDF5 / checkpoint / final eval paths match noise80.

F. If all relevant noise80 policy evals fail, the most likely and directly verified reason is: the learned policy reaches the handle area but does not produce the phase transition from approach to grasp and lift. It keeps grippers negative/open and z lift commands near zero/negative, so the pot is never raised enough for success. The upstream reason is likely the noise80 training distribution: base data is dominated by noisy trajectories/timesteps, while DAgger's added expert-labeled data is too small to overcome the mixed, conflicting phase-transition signal.

## Artifacts
- `diagnose_noise80.py`
- `static_audit.json`
- `static_audit_summary.txt`
- `offline_action_error.csv`
- `rollout_trace_summary_1ep.csv`
- `trace_noise50_r3_1ep.json`
- `trace_noise80_r1_1ep.json`
- `trace_noise80_r3_1ep.json`
- `trace_noise80_r4_1ep.json`
- `trace_noise100_r3_1ep.json`
- `trace_noise80_r3_3ep.json`
- `trace_noise100_r3_3ep.json`
- `trace_3ep_comparison.csv`
- `policy_vs_expert_rollout.py`
- `policy_vs_expert_rollout_3ep.csv`
- `policy_vs_expert_rollout_3ep.json`
- `expert_stage_diagnostics.py`
- `expert_stage_diagnostics_3ep.csv`
- `forced_grasp_lift.py`
- `forced_grasp_lift_noise80_r3_3ep.csv`

## Addendum: all noise80 rounds short-trace check
Source: `noise80_all_rounds_3ep_trace_summary.csv`

I also ran 3 short rollout traces for noise80 rounds 1, 2, 3, and 4. All 12 traced episodes failed. The failure pattern is consistent across all rounds:

- each round reaches the handle neighborhood in at least some episodes (`min_both_eef_handle_dist` often below 0.025);
- `grip0_close_frac = 0.0` and `grip1_close_frac = 0.0` for every traced episode;
- `first_both_grip_close_step` is never present;
- max gripper action remains negative in every traced episode;
- max upward z action remains small, never close to the successful lift values around 0.8;
- pot z increase remains near zero, except tiny incidental changes far below the success lift.

This strengthens the conclusion that the noise80 failure is not a one-round artifact. The shared closed-loop failure mode across rounds is missing grasp/lift transition after approach.
