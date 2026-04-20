"""
Training loop for the self-pruning network on CIFAR-10.

Runs a sweep over multiple lambda values to map the
sparsity-vs-accuracy trade-off.

Usage:
    python -m src.train                      # full sweep
    python -m src.train --lambdas 1e-4       # single value
    python -m src.train --epochs 5 --fast    # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from src.model import SelfPruningNet


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
@dataclass
class RunResult:
    lambda_val: float
    epochs: int
    final_train_loss: float
    final_cls_loss: float
    final_sparsity_loss: float
    test_accuracy: float
    global_sparsity_pct: float
    per_layer_sparsity: List[dict] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)  # per-epoch snapshots


# ---------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------- #
def build_loaders(batch_size: int, data_dir: str, num_workers: int = 2):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tfms
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------- #
# Train / eval primitives
# ---------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_one_run(
    lambda_val: float,
    epochs: int,
    lr: float,
    batch_size: int,
    data_dir: str,
    device: torch.device,
    num_workers: int = 2,
    log_every: int = 100,
    save_dir: Path | None = None,
) -> RunResult:
    """Train a single model at a given lambda; return full result."""
    print(f"\n{'=' * 68}")
    print(f"  λ = {lambda_val}  |  epochs = {epochs}  |  lr = {lr}")
    print(f"{'=' * 68}")

    train_loader, test_loader = build_loaders(batch_size, data_dir, num_workers)

    model = SelfPruningNet().to(device)

    # Split optimizers: Adam for weights, plain SGD for gate_scores.
    # Adam's per-parameter RMS normalization would cancel out the
    # sparsity gradient (near-uniform across gates at init), so SGD
    # is used for the gates where the update scales linearly with λ.
    # Gate LR is set absolutely rather than relative to weight LR —
    # gates need a much larger step size per update to travel the
    # ~6 units on the score axis required to hit sigmoid < 1e-2.
    weight_params = [p for n, p in model.named_parameters()
                     if "gate_scores" not in n]
    gate_params = [p for n, p in model.named_parameters()
                   if "gate_scores" in n]
    weight_opt = torch.optim.Adam(weight_params, lr=lr)
    gate_opt = torch.optim.SGD(gate_params, lr=0.02)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    warmup_epochs = 3  # ramp λ linearly during first N epochs

    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "sparse": 0.0, "n": 0}
        t0 = time.time()

        # Linear λ warmup gives the classifier time to establish which
        # weights are useful before the L1 penalty starts acting.
        if epoch <= warmup_epochs:
            current_lambda = lambda_val * (epoch / warmup_epochs)
        else:
            current_lambda = lambda_val

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(x)
            cls_loss = ce(logits, y)
            sp_loss = model.sparsity_loss()
            loss = cls_loss + current_lambda * sp_loss

            weight_opt.zero_grad(set_to_none=True)
            gate_opt.zero_grad(set_to_none=True)
            loss.backward()
            weight_opt.step()
            gate_opt.step()

            bs = y.size(0)
            running["total"] += loss.item() * bs
            running["cls"] += cls_loss.item() * bs
            running["sparse"] += sp_loss.item() * bs
            running["n"] += bs

            if step % log_every == 0:
                print(
                    f"  epoch {epoch:02d}  step {step:04d}  "
                    f"λ={current_lambda:.4f}  "
                    f"cls={cls_loss.item():.4f}  "
                    f"sparse={sp_loss.item():.1f}  "
                    f"total={loss.item():.4f}"
                )

        scheduler.step()

        # End-of-epoch metrics
        stats = model.sparsity_stats()
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - t0

        snapshot = {
            "epoch": epoch,
            "train_total_loss": running["total"] / running["n"],
            "train_cls_loss": running["cls"] / running["n"],
            "train_sparsity_loss": running["sparse"] / running["n"],
            "test_accuracy": test_acc,
            "global_sparsity_pct": stats["global_sparsity_pct"],
            "per_layer_sparsity_pct": [l["sparsity_pct"] for l in stats["per_layer"]],
            "epoch_time_sec": epoch_time,
        }
        history.append(snapshot)

        print(
            f"  [epoch {epoch:02d}] "
            f"test_acc={test_acc:5.2f}%  "
            f"sparsity={stats['global_sparsity_pct']:5.2f}%  "
            f"per_layer={[round(s,1) for s in snapshot['per_layer_sparsity_pct']]}  "
            f"({epoch_time:.1f}s)"
        )

    # Final stats
    final_stats = model.sparsity_stats()
    final_acc = evaluate(model, test_loader, device)

    result = RunResult(
        lambda_val=lambda_val,
        epochs=epochs,
        final_train_loss=history[-1]["train_total_loss"],
        final_cls_loss=history[-1]["train_cls_loss"],
        final_sparsity_loss=history[-1]["train_sparsity_loss"],
        test_accuracy=final_acc,
        global_sparsity_pct=final_stats["global_sparsity_pct"],
        per_layer_sparsity=final_stats["per_layer"],
        history=history,
    )

    # Save checkpoint + gate distribution
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = f"lambda_{lambda_val:.0e}".replace("+", "").replace("-0", "-")
        ckpt_path = save_dir / f"{tag}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "lambda_val": lambda_val,
            "result": asdict(result),
            "gate_values": model.all_gate_values(),
        }, ckpt_path)
        print(f"  saved checkpoint → {ckpt_path}")

    return result


# ---------------------------------------------------------------------- #
# Entrypoint
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[1e-5, 1e-4, 5e-4],
                        help="Lambda values to sweep (low/medium/high)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fast", action="store_true",
                        help="Smoke-test: use a tiny subset for a single epoch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for lam in args.lambdas:
        result = train_one_run(
            lambda_val=lam,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            device=device,
            num_workers=args.num_workers,
            save_dir=ckpt_dir,
        )
        all_results.append(asdict(result))

        # Dump incremental results after each run (so partial sweeps are usable)
        with open(results_dir / "sweep_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n\nSUMMARY")
    print(f"{'Lambda':>10}  {'Test Acc':>10}  {'Sparsity':>10}")
    print("-" * 36)
    for r in all_results:
        print(f"{r['lambda_val']:>10.0e}  "
              f"{r['test_accuracy']:>9.2f}%  "
              f"{r['global_sparsity_pct']:>9.2f}%")
    print(f"\nresults saved → {results_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
