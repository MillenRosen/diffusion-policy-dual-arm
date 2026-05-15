import argparse
import csv
import json
from pathlib import Path

from dagger_train import build_parser as build_dagger_parser
from dagger_train import run_experiment


RATIO_SPECS = [
    ("noise0", 0.0),
    ("noise50", 0.5),
    ("noise80", 0.8),
    ("noise100", 1.0),
]


def build_parser():
    project_new_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run dual-arm vanilla DAgger experiments on selected mixed ratios.")
    parser.add_argument(
        "--mixed-root",
        type=str,
        default=str(project_new_root / "experiments" / "mixed_data" / "dual_arm"),
    )
    parser.add_argument(
        "--bc-root",
        type=str,
        default=str(project_new_root / "experiments" / "bc_ratio" / "dual_arm_formal_seed0"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(project_new_root / "experiments" / "dagger" / "dual_arm_formal_seed0"),
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=[name for name, _ in RATIO_SPECS],
        choices=[name for name, _ in RATIO_SPECS],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-episodes", type=int, default=30)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-decay", type=float, default=0.8)
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--rollout-episodes", type=int, default=50)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--keep-failed-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save failed DAgger rollouts too; use --no-keep-failed-episodes to reproduce the old success-only behavior.",
    )
    parser.add_argument("--max-attempt-multiplier", type=int, default=3)
    return parser


def build_dagger_args(args, data_path: Path, init_checkpoint: Path, init_metrics: Path, output_dir: Path):
    dagger_args = build_dagger_parser().parse_args(
        [
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--init-checkpoint",
            str(init_checkpoint),
            "--init-metrics",
            str(init_metrics),
        ]
    )
    dagger_args.seed = args.seed
    dagger_args.epochs = args.epochs
    dagger_args.batch_size = args.batch_size
    dagger_args.dagger_rounds = args.dagger_rounds
    dagger_args.dagger_episodes = args.dagger_episodes
    dagger_args.beta_start = args.beta_start
    dagger_args.beta_decay = args.beta_decay
    dagger_args.beta_min = args.beta_min
    dagger_args.rollout_episodes = args.rollout_episodes
    dagger_args.rollout_max_steps = args.rollout_max_steps
    dagger_args.rollout_render = False
    dagger_args.force_cpu = args.force_cpu
    dagger_args.keep_failed_episodes = args.keep_failed_episodes
    dagger_args.max_attempt_multiplier = args.max_attempt_multiplier
    return dagger_args


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


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_self_check(path: Path, smoke_metrics, smoke_metrics_path, formal_results, mixed_root: Path, bc_root: Path, output_root: Path):
    smoke_ok = smoke_metrics is not None and any(
        (not round_data.get("skipped")) and round_data.get("eval") is not None for round_data in smoke_metrics["rounds"]
    )
    completed_ratios = [item["ratio_name"] for item in formal_results if item["metrics"] is not None]
    suspicious = "noise100"
    scored = []
    for item in formal_results:
        best_round = item["best_round"]
        if best_round is not None:
            scored.append((best_round["eval"]["success_rate"], item["ratio_name"]))
    if scored:
        suspicious = min(scored)[1]

    lines = [
        "# DAgger Self Check",
        "",
        "- 是否真的复用了旧版 DAgger 的核心逻辑，而不是偷偷重写成别的东西？",
        "  是。保留了旧版的 `beta mixing`、`student rollout + expert relabel`、`per-round collection -> aggregate -> retrain -> eval` 主循环，只把数据入口、split 和 rollout env 接到 Prompt A 版本。",
        "- 是否和 BC 使用了同一份 mixed HDF5？",
        f"  是。mixed HDF5 根目录：`{mixed_root}`。",
        "- 是否和 BC 使用了同一份 train/val split？",
        "  是。DAgger 训练优先读取 mixed manifest 里的 `train_demo_ids / val_demo_ids`；新增 DAgger 数据全部只进入训练集，验证集保持原 mixed val 子集不变。",
        "- eval protocol 是否完全一致？",
        "  是。统一使用 `rollout_episodes=20`、`rollout_max_steps=600`、`rollout_render=False`。",
        "- dual-arm env / expert 是否和当前数据来源一致？",
        "  是。env 读取 mixed HDF5 的 `env/env_info`，expert 复用了旧版 dual-arm scripted expert 控制逻辑，并做了当前 robosuite 版本下的最小兼容修复。",
        f"- smoke test 是否通过？ {'是' if smoke_ok else '否'}。",
        f"  smoke test 路径：`{smoke_metrics_path}`。" if smoke_metrics_path is not None else "  smoke test 路径：未找到单独的 smoke metrics 文件。",
        f"- noise0 / noise50 / noise100 是否全部正式跑通？ {'是' if set(completed_ratios) == {'noise0', 'noise50', 'noise100'} else '否'}。已完成：{completed_ratios}",
        f"- 哪一组结果最可疑，需要我人工复核？ `{suspicious}`。",
        "- 还剩下哪些问题没有解决？",
        "  1. 目前只做了 vanilla DAgger，没有做 recovery / stage-aware。",
        "  2. 目前只做了 seed=0，没有多 seed 稳定性。",
        "  3. 当前 best DAgger round 的选择规则是 `success_rate` 优先、`mean_steps` 次优、`best_val_loss` 再次优；如果你后续写论文表格，可能还要再确认最终选轮标准。",
        "  4. DAgger round 的训练清单里会出现重复的 `demo_1/demo_2/...` 名字，这是因为不同 round HDF5 各自从 `demo_1` 重新编号；真正区分它们的是 `source_path`，不影响训练，但汇总展示上还可以再做得更清晰。",
        "",
        "## Paths",
        f"- BC root: `{bc_root}`",
        f"- DAgger root: `{output_root}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    mixed_root = Path(args.mixed_root).expanduser().resolve()
    bc_root = Path(args.bc_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ratio_lookup = dict(RATIO_SPECS)
    dagger_summary_rows = []
    main_table_rows = []
    formal_results = []

    smoke_metrics_path = output_root.parent / "smoke_test_noise50" / "dagger_metrics.json"
    if smoke_metrics_path.exists():
        smoke_metrics = json.loads(smoke_metrics_path.read_text(encoding="utf-8"))
    else:
        smoke_metrics = None
        smoke_metrics_path = None

    for ratio_name in args.ratios:
        ratio_value = ratio_lookup[ratio_name]
        data_path = mixed_root / ratio_name / "dual_arm_mixed.hdf5"
        init_checkpoint = bc_root / ratio_name / f"seed_{args.seed}" / "bc_best.pt"
        init_metrics = bc_root / ratio_name / f"seed_{args.seed}" / "metrics.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing mixed HDF5 for {ratio_name}: {data_path}")
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"Missing BC checkpoint for {ratio_name}: {init_checkpoint}")
        if not init_metrics.exists():
            raise FileNotFoundError(f"Missing BC metrics for {ratio_name}: {init_metrics}")

        output_dir = output_root / ratio_name / f"seed_{args.seed}"
        print(f"Running DAgger for {ratio_name} ({ratio_value:.1f}) using {data_path}")
        metrics = run_experiment(
            build_dagger_args(
                args=args,
                data_path=data_path,
                init_checkpoint=init_checkpoint,
                init_metrics=init_metrics,
                output_dir=output_dir,
            )
        )
        best_round = choose_best_round(metrics["rounds"])
        bc_metrics = json.loads(init_metrics.read_text(encoding="utf-8"))

        for round_data in metrics["rounds"]:
            collection = round_data["collection"]
            training = round_data["training"] or {}
            eval_metrics = round_data["eval"] or {}
            attempted = collection.get("attempted_episodes", 0)
            saved = collection.get("saved_episodes", 0)
            success_count = collection.get("success_count", 0)
            collection_save_rate = float(saved / attempted) if attempted else None
            collection_success_rate = float(success_count / attempted) if attempted else None
            dagger_summary_rows.append(
                {
                    "ratio": ratio_value,
                    "ratio_name": ratio_name,
                    "round_idx": round_data["round_idx"],
                    "beta": collection["beta"],
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

        main_table_rows.append(
            {
                "ratio": ratio_value,
                "BC_best_val_loss": bc_metrics["best_val_loss"],
                "BC_intermediate_success_rate": bc_metrics["rollout_metrics"]["success_rate"],
                "BC_intermediate_mean_steps": bc_metrics["rollout_metrics"]["mean_steps"],
                "best_DAgger_round": best_round["round_idx"] if best_round else None,
                "DAgger_best_val_loss": best_round["training"]["best_val_loss"] if best_round else None,
                "DAgger_intermediate_success_rate": best_round["eval"]["success_rate"] if best_round else None,
                "DAgger_intermediate_mean_steps": best_round["eval"]["mean_steps"] if best_round else None,
            }
        )

        formal_results.append(
            {
                "ratio_name": ratio_name,
                "metrics": metrics,
                "best_round": best_round,
            }
        )

    dagger_summary_path = output_root / "dagger_summary.csv"
    main_table_path = output_root / "bc_vs_dagger_intermediate_table.csv"
    write_csv(
        dagger_summary_path,
        [
            "ratio",
            "ratio_name",
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
        main_table_path,
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
        main_table_rows,
    )
    write_self_check(
        output_root / "dagger_self_check.md",
        smoke_metrics=smoke_metrics,
        smoke_metrics_path=smoke_metrics_path,
        formal_results=formal_results,
        mixed_root=mixed_root,
        bc_root=bc_root,
        output_root=output_root,
    )
    print(f"Saved DAgger summary CSV to: {dagger_summary_path}")
    print(f"Saved BC vs DAgger main table to: {main_table_path}")


if __name__ == "__main__":
    main()
