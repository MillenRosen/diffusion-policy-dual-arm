# Excluded Large Files

This repository intentionally excludes large or non-essential experiment artifacts so it can be uploaded cleanly to GitHub.

## Excluded File Types

- Demonstration datasets: `*.hdf5`, `*.h5`
- NumPy array dumps: `*.npz`, `*.npy`
- Model checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- Large videos: `*.mp4`, `*.avi`, `*.mov`
- Temporary logs and caches: `*.log`, `__pycache__/`, `.ipynb_checkpoints/`, `wandb/`, `runs/`, `tmp/`

## Excluded Experiment Trees

- Large `dagger_round_*` rollout folders
- Repeated checkpoint directories
- Intermediate `mixed_data/` folders
- Raw demonstrations folders
- Large trace-heavy diagnostics that are not needed for a concise GitHub submission

## Reason

Large demonstration datasets and trained checkpoints are excluded due to file size limits. They can be provided separately if required.
