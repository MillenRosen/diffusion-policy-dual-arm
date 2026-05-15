import csv
import json
from pathlib import Path

import torch

from bc_train import run_rollout_evaluation, set_seed
from dagger_train import load_policy


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def eval_checkpoint(label, checkpoint_path, data_path, episodes, max_steps, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(checkpoint_path, device=device)
    metrics = run_rollout_evaluation(
        model=model,
        checkpoint=checkpoint,
        source_hdf5_path=data_path,
        rollout_episodes=episodes,
        rollout_max_steps=max_steps,
        rollout_render=False,
        device=device,
    )
    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "success_rate": metrics["success_rate"],
        "success_count": metrics["success_count"],
        "episodes": metrics["episodes"],
        "mean_steps": metrics["mean_steps"],
        "mean_reward": metrics["mean_reward"],
        "metrics": metrics,
    }


def main():
    final_root = Path(r"D:\A-6019\project\project_new\experiments\final_dual_arm")
    data_path = final_root / "mixed_data" / "noise0" / "dual_arm_mixed.hdf5"
    output_root = final_root / "sanity_eval_50ep_noise0"
    episodes = 50
    max_steps = 600
    seed = 0

    jobs = [
        (
            "BC_noise0",
            final_root / "bc_ratio" / "noise0" / "seed_0" / "bc_best.pt",
        ),
        (
            "DAgger_noise0_round2",
            final_root / "dagger" / "noise0" / "seed_0" / "round_2_train" / "dagger_round_2_best.pt",
        ),
    ]

    rows = []
    for label, checkpoint_path in jobs:
        result = eval_checkpoint(label, checkpoint_path, data_path, episodes, max_steps, seed)
        write_json(output_root / f"{label}.json", result["metrics"])
        rows.append(
            {
                "label": label,
                "checkpoint_path": result["checkpoint_path"],
                "success_rate": result["success_rate"],
                "success_count": result["success_count"],
                "episodes": result["episodes"],
                "mean_steps": result["mean_steps"],
                "mean_reward": result["mean_reward"],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "checkpoint_path",
                "success_rate",
                "success_count",
                "episodes",
                "mean_steps",
                "mean_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
