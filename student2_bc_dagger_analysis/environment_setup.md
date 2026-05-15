# Environment Setup

This repository is a lightweight packaging of the Student 2 code and results. Full reruns require access to the original datasets and trained checkpoints, which are not included here.

## Suggested Python Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Runtime Notes

- The training and evaluation scripts use PyTorch, NumPy, h5py, matplotlib, and robosuite.
- Rollout evaluation requires a robosuite-compatible simulator environment.
- The original experiments used large dual-arm HDF5 demonstration datasets and saved checkpoints that are excluded from this GitHub-ready package.
- Reproduction commands should be run with paths that point to separately supplied demonstrations, checkpoints, and experiment-output directories.
