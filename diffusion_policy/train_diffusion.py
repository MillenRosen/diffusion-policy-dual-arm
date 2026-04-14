from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合脚本运行
import matplotlib.pyplot as plt

from diffusion import DiffusionScheduler
from model import ConditionalUnet1D


def flatten_obs(obs_batch: torch.Tensor) -> torch.Tensor:
    return obs_batch.reshape(obs_batch.shape[0], -1)


def add_noise(data, noise_std=0.005):
    return data + torch.randn_like(data) * noise_std


def run_epoch(model, scheduler, loader, optimizer, device, train=True):
    criterion = nn.MSELoss()
    total_loss = 0.0
    if train:
        model.train()
    else:
        model.eval()

    for obs_batch, act_batch in loader:
        obs_batch = flatten_obs(obs_batch.to(device))
        act_batch = act_batch.to(device)
        t = torch.randint(0, scheduler.num_steps, (obs_batch.shape[0],), device=device)
        noise = torch.randn_like(act_batch)
        noisy_act = scheduler.q_sample(act_batch, t, noise)

        with torch.set_grad_enabled(train):
            pred_noise = model(noisy_act, t, obs_batch)
            loss = criterion(pred_noise, noise)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * obs_batch.shape[0]
    return total_loss / len(loader.dataset)


def plot_loss_curve(history, output_dir):
    """绘制训练和验证损失曲线，并保存为 loss_curve.png"""
    epochs = range(1, len(history["train_losses"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_losses"], label="Train Loss", marker='o', markersize=3, linewidth=1)
    plt.plot(epochs, history["val_losses"], label="Validation Loss", marker='s', markersize=3, linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")  # 损失通常跨度较大，对数坐标更清晰
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = Path(output_dir) / "loss_curve.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "processed_scripted_c2" / "diffusion_h2_c2_data.npz"),
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "processed_scripted_c2" / "norm_stats.npz"),
    )
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "checkpoints_c2"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--cond_dim", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--down_dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--n_groups", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--data_augment", action="store_true")
    parser.add_argument("--noise_std", type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    obs_train = torch.from_numpy(data["obs_train"]).float()
    act_train = torch.from_numpy(data["act_train"]).float()
    obs_val = torch.from_numpy(data["obs_val"]).float()
    act_val = torch.from_numpy(data["act_val"]).float()

    if args.data_augment:
        obs_train = add_noise(obs_train, args.noise_std)
        act_train = add_noise(act_train, args.noise_std)

    train_loader = DataLoader(
        TensorDataset(obs_train, act_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        TensorDataset(obs_val, act_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    obs_dim = obs_train.shape[1] * obs_train.shape[2]
    action_chunk = act_train.shape[1]
    act_dim = act_train.shape[2]

    model = ConditionalUnet1D(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_chunk=action_chunk,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        down_dims=args.down_dims,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        dropout=args.dropout,
    ).to(device)
    scheduler = DiffusionScheduler(num_steps=args.num_diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = {"train_losses": [], "val_losses": [], "best_val_loss": None}
    best_val_loss = float("inf")
    patience = 0
    best_path = output_dir / "best_diffusion_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, scheduler, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, scheduler, val_loader, optimizer, device, train=False)
        lr_scheduler.step(val_loss)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_val_loss"] = val_loss
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "obs_horizon": int(obs_train.shape[1]),
                    "obs_dim": int(obs_train.shape[2]),
                    "flat_obs_dim": int(obs_dim),
                    "act_dim": int(act_dim),
                    "action_chunk": int(action_chunk),
                    "norm_stats_path": args.norm_stats_path,
                    "data_path": args.data_path,
                    "val_loss": float(val_loss),
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 保存训练历史到 JSON
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best val loss: {best_val_loss:.6f}")

    # 绘制损失曲线
    plot_loss_curve(history, output_dir)


if __name__ == "__main__":
    main()