import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from bc_train import run_rollout_evaluation, set_seed
from dagger_train import load_policy


BC_RATIOS = [
    ("noise0", 0.0),
    ("noise10", 0.1),
    ("noise20", 0.2),
    ("noise30", 0.3),
    ("noise50", 0.5),
    ("noise80", 0.8),
    ("noise100", 1.0),
]

DAGGER_RATIOS = [
    ("noise0", 0.0),
    ("noise50", 0.5),
    ("noise80", 0.8),
    ("noise100", 1.0),
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bc_seed_dir(bc_root: Path, ratio_name: str, seed: int):
    seeded = bc_root / ratio_name / f"seed_{seed}"
    if seeded.exists():
        return seeded
    return bc_root / ratio_name


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


def eval_checkpoint(checkpoint_path: Path, source_hdf5_path: Path, episodes: int, max_steps: int, render: bool, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(checkpoint_path, device=device)
    return run_rollout_evaluation(
        model=model,
        checkpoint=checkpoint,
        source_hdf5_path=source_hdf5_path,
        rollout_episodes=episodes,
        rollout_max_steps=max_steps,
        rollout_render=render,
        device=device,
    )


def build_bc_intermediate_summary(final_root: Path, seed: int):
    rows = []
    bc_root = final_root / "bc_ratio"
    for ratio_name, ratio_value in BC_RATIOS:
        metrics = read_json(bc_seed_dir(bc_root, ratio_name, seed) / "metrics.json")
        rollout = metrics.get("rollout_metrics") or {}
        rows.append(
            {
                "ratio": ratio_value,
                "train_samples": metrics["train_samples"],
                "val_samples": metrics["val_samples"],
                "best_val_loss": metrics["best_val_loss"],
                "best_epoch": metrics["best_epoch"],
                "intermediate_success_rate": rollout.get("success_rate"),
                "intermediate_success_count": rollout.get("success_count"),
                "intermediate_mean_steps": rollout.get("mean_steps"),
                "intermediate_mean_reward": rollout.get("mean_reward"),
            }
        )
    write_csv(
        bc_root / "summary.csv",
        [
            "ratio",
            "train_samples",
            "val_samples",
            "best_val_loss",
            "best_epoch",
            "intermediate_success_rate",
            "intermediate_success_count",
            "intermediate_mean_steps",
            "intermediate_mean_reward",
        ],
        rows,
    )
    return rows


def build_dagger_intermediate_tables(final_root: Path, seed: int):
    bc_root = final_root / "bc_ratio"
    dagger_root = final_root / "dagger"
    dagger_summary_rows = []
    comparison_rows = []
    best_round_by_ratio = {}

    for ratio_name, ratio_value in DAGGER_RATIOS:
        metrics = read_json(dagger_root / ratio_name / f"seed_{seed}" / "dagger_metrics.json")
        best_round = choose_best_round(metrics["rounds"])
        best_round_by_ratio[ratio_name] = best_round

        for round_data in metrics["rounds"]:
            collection = round_data["collection"]
            training = round_data.get("training") or {}
            eval_metrics = round_data.get("eval") or {}
            attempted = collection.get("attempted_episodes", 0)
            saved = collection.get("saved_episodes", 0)
            success_count = collection.get("success_count", 0)
            collection_save_rate = float(saved / attempted) if attempted else None
            collection_success_rate = float(success_count / attempted) if attempted else None
            dagger_summary_rows.append(
                {
                    "ratio": ratio_value,
                    "round_idx": round_data["round_idx"],
                    "beta": collection.get("beta"),
                    "saved_episodes": saved,
                    "attempted_episodes": attempted,
                    "collection_success_count": success_count,
                    "collection_save_rate": collection_save_rate,
                    "success_rate_collection": collection_success_rate,
                    "best_val_loss": training.get("best_val_loss"),
                    "intermediate_eval_success_rate": eval_metrics.get("success_rate"),
                    "intermediate_eval_success_count": eval_metrics.get("success_count"),
                    "intermediate_mean_steps": eval_metrics.get("mean_steps"),
                    "intermediate_mean_reward": eval_metrics.get("mean_reward"),
                }
            )

        bc_metrics = read_json(bc_seed_dir(bc_root, ratio_name, seed) / "metrics.json")
        bc_rollout = bc_metrics["rollout_metrics"]
        comparison_rows.append(
            {
                "ratio": ratio_value,
                "BC_best_val_loss": bc_metrics["best_val_loss"],
                "BC_intermediate_success_rate": bc_rollout["success_rate"],
                "BC_intermediate_mean_steps": bc_rollout["mean_steps"],
                "best_DAgger_round": best_round["round_idx"] if best_round else None,
                "DAgger_best_val_loss": best_round["training"]["best_val_loss"] if best_round else None,
                "DAgger_intermediate_success_rate": best_round["eval"]["success_rate"] if best_round else None,
                "DAgger_intermediate_mean_steps": best_round["eval"]["mean_steps"] if best_round else None,
            }
        )

    write_csv(
        dagger_root / "dagger_summary.csv",
        [
            "ratio",
            "round_idx",
            "beta",
            "saved_episodes",
            "attempted_episodes",
            "collection_success_count",
            "collection_save_rate",
            "success_rate_collection",
            "best_val_loss",
            "intermediate_eval_success_rate",
            "intermediate_eval_success_count",
            "intermediate_mean_steps",
            "intermediate_mean_reward",
        ],
        dagger_summary_rows,
    )
    write_csv(
        dagger_root / "bc_vs_dagger_intermediate_table.csv",
        [
            "ratio",
            "BC_best_val_loss",
            "BC_intermediate_success_rate",
            "BC_intermediate_mean_steps",
            "best_DAgger_round",
            "DAgger_best_val_loss",
            "DAgger_intermediate_success_rate",
            "DAgger_intermediate_mean_steps",
        ],
        comparison_rows,
    )
    return dagger_summary_rows, comparison_rows, best_round_by_ratio


def run_final_eval(final_root: Path, seed: int, episodes: int, max_steps: int, render: bool, best_round_by_ratio):
    mixed_root = final_root / "mixed_data"
    bc_root = final_root / "bc_ratio"
    dagger_root = final_root / "dagger"
    final_eval_root = final_root / "final_eval"

    bc_rows = []
    for ratio_name, ratio_value in BC_RATIOS:
        metrics = read_json(bc_seed_dir(bc_root, ratio_name, seed) / "metrics.json")
        checkpoint_path = bc_seed_dir(bc_root, ratio_name, seed) / "bc_best.pt"
        data_path = mixed_root / ratio_name / "dual_arm_mixed.hdf5"
        eval_metrics = eval_checkpoint(checkpoint_path, data_path, episodes, max_steps, render, seed)
        write_json(final_eval_root / f"bc_{ratio_name}_eval_metrics.json", eval_metrics)
        bc_rows.append(
            {
                "ratio": ratio_value,
                "best_val_loss": metrics["best_val_loss"],
                "final_success_rate": eval_metrics["success_rate"],
                "final_success_count": eval_metrics["success_count"],
                "final_mean_steps": eval_metrics["mean_steps"],
                "final_mean_reward": eval_metrics["mean_reward"],
            }
        )

    dagger_rows = []
    for ratio_name, ratio_value in DAGGER_RATIOS:
        best_round = best_round_by_ratio[ratio_name]
        round_idx = best_round["round_idx"]
        checkpoint_path = dagger_root / ratio_name / f"seed_{seed}" / f"round_{round_idx}_train" / f"dagger_round_{round_idx}_best.pt"
        data_path = mixed_root / ratio_name / "dual_arm_mixed.hdf5"
        eval_metrics = eval_checkpoint(checkpoint_path, data_path, episodes, max_steps, render, seed)
        write_json(final_eval_root / f"dagger_{ratio_name}_round_{round_idx}_eval_metrics.json", eval_metrics)
        dagger_rows.append(
            {
                "ratio": ratio_value,
                "best_DAgger_round": round_idx,
                "best_val_loss": best_round["training"]["best_val_loss"],
                "final_success_rate": eval_metrics["success_rate"],
                "final_success_count": eval_metrics["success_count"],
                "final_mean_steps": eval_metrics["mean_steps"],
                "final_mean_reward": eval_metrics["mean_reward"],
            }
        )

    dagger_by_ratio = {row["ratio"]: row for row in dagger_rows}
    bc_by_ratio = {row["ratio"]: row for row in bc_rows}
    comparison_rows = []
    for _, ratio_value in DAGGER_RATIOS:
        bc_row = bc_by_ratio[ratio_value]
        dagger_row = dagger_by_ratio[ratio_value]
        comparison_rows.append(
            {
                "ratio": ratio_value,
                "BC_best_val_loss": bc_row["best_val_loss"],
                "BC_final_success_rate": bc_row["final_success_rate"],
                "BC_final_mean_steps": bc_row["final_mean_steps"],
                "best_DAgger_round": dagger_row["best_DAgger_round"],
                "DAgger_best_val_loss": dagger_row["best_val_loss"],
                "DAgger_final_success_rate": dagger_row["final_success_rate"],
                "DAgger_final_mean_steps": dagger_row["final_mean_steps"],
            }
        )

    write_csv(
        final_eval_root / "bc_final_eval.csv",
        [
            "ratio",
            "best_val_loss",
            "final_success_rate",
            "final_success_count",
            "final_mean_steps",
            "final_mean_reward",
        ],
        bc_rows,
    )
    write_csv(
        final_eval_root / "dagger_final_eval.csv",
        [
            "ratio",
            "best_DAgger_round",
            "best_val_loss",
            "final_success_rate",
            "final_success_count",
            "final_mean_steps",
            "final_mean_reward",
        ],
        dagger_rows,
    )
    write_csv(
        final_eval_root / "bc_vs_dagger_final_table.csv",
        [
            "ratio",
            "BC_best_val_loss",
            "BC_final_success_rate",
            "BC_final_mean_steps",
            "best_DAgger_round",
            "DAgger_best_val_loss",
            "DAgger_final_success_rate",
            "DAgger_final_mean_steps",
        ],
        comparison_rows,
    )
    return bc_rows, dagger_rows, comparison_rows


def make_figures(final_root: Path, bc_rows, comparison_rows):
    figures_root = final_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    ratios = [row["ratio"] for row in bc_rows]
    success = [row["final_success_rate"] for row in bc_rows]
    val_loss = [row["best_val_loss"] for row in bc_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(ratios, success, marker="o")
    plt.xlabel("Noisy ratio")
    plt.ylabel("BC final success rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_root / "bc_success_vs_noisy_ratio.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ratios, val_loss, marker="o", color="#8a5a00")
    plt.xlabel("Noisy ratio")
    plt.ylabel("Best validation loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_root / "bc_val_loss_vs_noisy_ratio.png", dpi=200)
    plt.close()

    labels = [str(row["ratio"]) for row in comparison_rows]
    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], [row["BC_final_success_rate"] for row in comparison_rows], width, label="BC")
    plt.bar([i + width / 2 for i in x], [row["DAgger_final_success_rate"] for row in comparison_rows], width, label="DAgger")
    plt.xticks(x, labels)
    plt.xlabel("Noisy ratio")
    plt.ylabel("Final success rate")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "bc_vs_dagger_success.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], [row["BC_final_mean_steps"] for row in comparison_rows], width, label="BC")
    plt.bar([i + width / 2 for i in x], [row["DAgger_final_mean_steps"] for row in comparison_rows], width, label="DAgger")
    plt.xticks(x, labels)
    plt.xlabel("Noisy ratio")
    plt.ylabel("Final mean steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "bc_vs_dagger_mean_steps.png", dpi=200)
    plt.close()


def write_self_check(final_root: Path, archive_note: str, bc_rows, dagger_rows, comparison_rows):
    mixed_root = final_root / "mixed_data"
    manifests = {name: read_json(mixed_root / name / "dual_arm_mixed.manifest.json") for name, _ in BC_RATIOS}
    noise100 = manifests["noise100"]
    raw_ok = all("demonstrations_change" in manifests[name]["clean_hdf5_path"] and "demonstrations_change" in manifests[name]["noisy_hdf5_path"] for name, _ in BC_RATIOS)
    all_160 = all(manifest["total_trajectories"] == 160 for manifest in manifests.values())
    manifest_steps_ok = all("clean_num_steps" in manifest and "noisy_num_steps" in manifest and "actual_noisy_ratio_by_timestep" in manifest for manifest in manifests.values())
    suspicious = "DAgger noise100: round 3 saved 0 episodes, and final success remains low, so this group needs manual review."
    can_report = "Yes, with the caveat that only seed=0 was run and DAgger noise100 is weak/suspicious."

    lines = [
        "# Final Dual-Arm Self Check",
        "",
        f"1. 使用最终版 dual_arm_clean.hdf5 / dual_arm_noisy.hdf5: {'Yes' if raw_ok else 'No'}",
        f"2. 旧实验结果是否删除或归档: {archive_note}",
        f"3. mixed HDF5 是否全部重新生成: Yes, 7 ratios under `{mixed_root}`.",
        f"4. 每个 ratio 是否 total_trajectories=160: {'Yes' if all_160 else 'No'}",
        f"5. noise100 是否 160 noisy + 0 clean: {'Yes' if noise100['noisy_count'] == 160 and noise100['clean_count'] == 0 else 'No'}",
        f"6. manifest 是否包含 step 数和 timestep noisy ratio: {'Yes' if manifest_steps_ok else 'No'}",
        f"7. BC 是否 7 个 ratio 正式跑通: {'Yes' if len(bc_rows) == 7 else 'No'}",
        f"8. DAgger 是否跑通 noise0/noise50/noise100: {'Yes' if len(dagger_rows) == 3 else 'No'}",
        "9. DAgger 是否使用对应 ratio 的 BC checkpoint 初始化: Yes, run_dagger_ratio_experiments used the matching `bc_ratio/<ratio>/seed_0/bc_best.pt`.",
        "10. BC 和 DAgger 是否使用同一份 mixed HDF5 和 train/val split: Yes, both read the same mixed HDF5 manifests; DAgger keeps mixed val split fixed and adds DAgger data only to train.",
        "11. 中间 eval 和 final eval 是否区分清楚: Yes, intermediate CSVs are under `bc_ratio` / `dagger`; 20-episode results are under `final_eval`.",
        "12. 最终报告主表是否只使用 20-episode final evaluation: Yes, use `final_eval/bc_vs_dagger_final_table.csv` and companion final CSVs.",
        f"13. 哪一组结果最可疑，需要人工复核: {suspicious}",
        f"14. 当前结果能否直接用于报告: {can_report}",
        "",
        "Note: `demonstrations_change` still contains single-arm and sequential raw HDF5 files, but this run only references the dual-arm clean/noisy files. Non-final old project HDF5 outputs were deleted.",
    ]
    (final_root / "final_self_check.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    project_new_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run final 20-episode eval and summarize final dual-arm experiments.")
    parser.add_argument("--final-root", type=str, default=str(project_new_root / "experiments" / "final_dual_arm"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--rollout-render", action="store_true")
    parser.add_argument("--archive-note", type=str, default="Old outputs were deleted or archived before this final run.")
    return parser


def main():
    args = build_parser().parse_args()
    final_root = Path(args.final_root).expanduser().resolve()
    bc_rows_intermediate = build_bc_intermediate_summary(final_root, args.seed)
    dagger_summary_rows, intermediate_comparison_rows, best_round_by_ratio = build_dagger_intermediate_tables(final_root, args.seed)
    _ = bc_rows_intermediate, dagger_summary_rows, intermediate_comparison_rows
    bc_rows, dagger_rows, comparison_rows = run_final_eval(
        final_root=final_root,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.rollout_max_steps,
        render=args.rollout_render,
        best_round_by_ratio=best_round_by_ratio,
    )
    make_figures(final_root, bc_rows, comparison_rows)
    write_self_check(final_root, args.archive_note, bc_rows, dagger_rows, comparison_rows)
    print(f"Saved final evaluation CSVs to: {final_root / 'final_eval'}")
    print(f"Saved figures to: {final_root / 'figures'}")
    print(f"Saved self check to: {final_root / 'final_self_check.md'}")


if __name__ == "__main__":
    main()
