import csv
import json
from pathlib import Path

import torch

from bc_train import run_rollout_evaluation, set_seed
from dagger_train import load_policy


RATIOS = ["noise0", "noise50", "noise80", "noise100"]
RATIO_VALUES = {"noise0": 0.0, "noise50": 0.5, "noise80": 0.8, "noise100": 1.0}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def eval_checkpoint(checkpoint_path: Path, data_path: Path, episodes: int, max_steps: int, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(checkpoint_path, device=device)
    return run_rollout_evaluation(
        model=model,
        checkpoint=checkpoint,
        source_hdf5_path=data_path,
        rollout_episodes=episodes,
        rollout_max_steps=max_steps,
        rollout_render=False,
        device=device,
    )


def choose_best_round(rounds):
    valid_rounds = [
        round_data
        for round_data in rounds
        if (
            not round_data.get("skipped")
            and round_data.get("training") is not None
            and round_data.get("eval") is not None
            and round_data["eval"].get("success_rate") is not None
        )
    ]
    if not valid_rounds:
        return None
    return max(
        valid_rounds,
        key=lambda round_data: (
            round_data["eval"]["success_rate"],
            -round_data["eval"]["mean_steps"],
            -round_data["training"]["best_val_loss"],
        ),
    )


def ratio_name_from_value(value: str):
    return f"noise{int(float(value) * 100)}"


def main():
    final_root = Path(r"D:\A-6019\project\project_new\experiments\final_dual_arm")
    output_root = final_root / "final_eval_50ep"
    episodes = 50
    max_steps = 600
    seed = 0

    dagger_final = read_csv(final_root / "final_eval" / "dagger_final_eval.csv")
    best_round_by_ratio = {
        ratio_name_from_value(row["ratio"]): int(row["best_DAgger_round"])
        for row in dagger_final
    }
    for ratio_name in RATIOS:
        if ratio_name not in best_round_by_ratio:
            metrics_path = final_root / "dagger" / ratio_name / "seed_0" / "dagger_metrics.json"
            metrics = read_json(metrics_path)
            best_round = choose_best_round(metrics["rounds"])
            if best_round is None:
                raise RuntimeError(f"No valid DAgger round found for {ratio_name}: {metrics_path}")
            best_round_by_ratio[ratio_name] = int(best_round["round_idx"])

    jobs = []
    for ratio_name in RATIOS:
        data_path = final_root / "mixed_data" / ratio_name / "dual_arm_mixed.hdf5"
        jobs.append(
            {
                "ratio_name": ratio_name,
                "ratio": RATIO_VALUES[ratio_name],
                "method": "BC",
                "round": "",
                "checkpoint_path": final_root / "bc_ratio" / ratio_name / "seed_0" / "bc_best.pt",
                "data_path": data_path,
                "metrics_path": output_root / f"bc_{ratio_name}_50ep.json",
            }
        )
        round_idx = best_round_by_ratio[ratio_name]
        jobs.append(
            {
                "ratio_name": ratio_name,
                "ratio": RATIO_VALUES[ratio_name],
                "method": "DAgger",
                "round": round_idx,
                "checkpoint_path": final_root
                / "dagger"
                / ratio_name
                / "seed_0"
                / f"round_{round_idx}_train"
                / f"dagger_round_{round_idx}_best.pt",
                "data_path": data_path,
                "metrics_path": output_root / f"dagger_{ratio_name}_round_{round_idx}_50ep.json",
            }
        )

    rows = []
    for job in jobs:
        if job["metrics_path"].exists():
            metrics = read_json(job["metrics_path"])
        elif job["ratio_name"] == "noise0" and job["method"] == "BC" and (
            final_root / "sanity_eval_50ep_noise0" / "BC_noise0.json"
        ).exists():
            metrics = read_json(final_root / "sanity_eval_50ep_noise0" / "BC_noise0.json")
            write_json(job["metrics_path"], metrics)
        elif job["ratio_name"] == "noise0" and job["method"] == "DAgger" and (
            final_root / "sanity_eval_50ep_noise0" / "DAgger_noise0_round2.json"
        ).exists():
            metrics = read_json(final_root / "sanity_eval_50ep_noise0" / "DAgger_noise0_round2.json")
            write_json(job["metrics_path"], metrics)
        else:
            metrics = eval_checkpoint(
                checkpoint_path=job["checkpoint_path"],
                data_path=job["data_path"],
                episodes=episodes,
                max_steps=max_steps,
                seed=seed,
            )
            write_json(job["metrics_path"], metrics)

        rows.append(
            {
                "ratio": job["ratio"],
                "ratio_name": job["ratio_name"],
                "method": job["method"],
                "best_DAgger_round": job["round"],
                "success_rate_50ep": metrics["success_rate"],
                "success_count_50ep": metrics["success_count"],
                "episodes": metrics["episodes"],
                "mean_steps_50ep": metrics["mean_steps"],
                "mean_reward_50ep": metrics["mean_reward"],
                "checkpoint_path": str(job["checkpoint_path"]),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "bc_vs_dagger_50ep_with_noise80_detailed.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "ratio_name",
                "method",
                "best_DAgger_round",
                "success_rate_50ep",
                "success_count_50ep",
                "episodes",
                "mean_steps_50ep",
                "mean_reward_50ep",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    comparison_rows = []
    by_key = {(row["ratio_name"], row["method"]): row for row in rows}
    for ratio_name in RATIOS:
        bc = by_key[(ratio_name, "BC")]
        dagger = by_key[(ratio_name, "DAgger")]
        comparison_rows.append(
            {
                "ratio": RATIO_VALUES[ratio_name],
                "BC_success_rate_50ep": bc["success_rate_50ep"],
                "BC_success_count_50ep": bc["success_count_50ep"],
                "BC_mean_steps_50ep": bc["mean_steps_50ep"],
                "best_DAgger_round": dagger["best_DAgger_round"],
                "DAgger_success_rate_50ep": dagger["success_rate_50ep"],
                "DAgger_success_count_50ep": dagger["success_count_50ep"],
                "DAgger_mean_steps_50ep": dagger["mean_steps_50ep"],
            }
        )

    comparison_path = output_root / "bc_vs_dagger_50ep_with_noise80.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "BC_success_rate_50ep",
                "BC_success_count_50ep",
                "BC_mean_steps_50ep",
                "best_DAgger_round",
                "DAgger_success_rate_50ep",
                "DAgger_success_count_50ep",
                "DAgger_mean_steps_50ep",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(json.dumps(comparison_rows, indent=2))
    print(f"Saved: {table_path}")
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
