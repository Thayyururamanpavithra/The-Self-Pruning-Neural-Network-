"""
Demo training runner (uses synthetic CIFAR-like data).

This is the offline/sandbox companion to src.train. It imports the
exact same model and training logic, but swaps in the synthetic dataset
because the sandbox cannot download real CIFAR-10.

On a machine with internet, run `python -m src.train` instead — it will
download real CIFAR-10 via torchvision.datasets.CIFAR10 and use the same
training loop.

The default lambda sweep (0.05, 0.25, 0.45) produces a clean low/
medium/high accuracy-vs-sparsity trade-off with these calibrated
hyperparameters:
  - weight optimizer: Adam, lr=1e-3
  - gate optimizer:   SGD,  lr=0.02 (separate, see train.py for why)
  - warmup: 3 epochs of linearly ramped lambda
  - epochs: 12
  - batch size: 256
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import SelfPruningNet
from src.synth_data import SynthCIFAR, make_synthetic_cifar
from src.train import RunResult, evaluate


def build_synth_loaders(batch_size, data_dir="./data/synth"):
    if not (Path(data_dir) / "train.npz").exists():
        make_synthetic_cifar(out_dir=data_dir)
    train = SynthCIFAR(f"{data_dir}/train.npz")
    test = SynthCIFAR(f"{data_dir}/test.npz")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=0, drop_last=True),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
    )


def train_one_run_synth(
    lambda_val: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    gate_lr: float = 0.02,
    warmup_epochs: int = 3,
    sparsity_form: str = "relu_shift",
    save_dir: Path | None = None,
) -> RunResult:
    print(f"\n{'=' * 72}")
    print(f"  λ = {lambda_val}  |  epochs = {epochs}  |  "
          f"lr = {lr}  |  gate_lr = {gate_lr}  |  warmup = {warmup_epochs}")
    print(f"{'=' * 72}")

    train_loader, test_loader = build_synth_loaders(batch_size)
    torch.manual_seed(42)          # reproducible across lambda sweep
    model = SelfPruningNet().to(device)

    # Split optimizers. Adam for weights, plain SGD for gate_scores:
    #   1. Adam's per-parameter RMS normalization washes out the nearly-
    #      uniform sparsity gradient, so gates barely move — you need SGD
    #      for the sparsity signal to actually propagate.
    #   2. Gates need a LR ~20x the weights' LR because the sparsity loss
    #      gradient is small on the score scale (bounded by 1 for
    #      relu_shift, by 0.25 for sigmoid) and gates need to travel
    #      ~6 units on the score axis to get sigmoid(s) < 1e-2.
    weight_params = [p for n, p in model.named_parameters()
                     if "gate_scores" not in n]
    gate_params = [p for n, p in model.named_parameters()
                   if "gate_scores" in n]
    weight_opt = torch.optim.Adam(weight_params, lr=lr)
    gate_opt = torch.optim.SGD(gate_params, lr=gate_lr)

    ce = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        # Linear lambda warmup: give CE a few epochs to establish which
        # weights matter before pruning pressure ramps up.
        if epoch <= warmup_epochs:
            current_lambda = lambda_val * (epoch / warmup_epochs)
        else:
            current_lambda = lambda_val

        model.train()
        sums = {"total": 0.0, "cls": 0.0, "sparse": 0.0, "n": 0}
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            cls_loss = ce(logits, y)
            sp_loss = model.sparsity_loss(form=sparsity_form)
            loss = cls_loss + current_lambda * sp_loss

            weight_opt.zero_grad(set_to_none=True)
            gate_opt.zero_grad(set_to_none=True)
            loss.backward()
            weight_opt.step()
            gate_opt.step()

            bs = y.size(0)
            sums["total"] += loss.item() * bs
            sums["cls"] += cls_loss.item() * bs
            sums["sparse"] += sp_loss.item() * bs
            sums["n"] += bs

        stats = model.sparsity_stats()
        test_acc = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch,
            "current_lambda": current_lambda,
            "train_total_loss": sums["total"] / sums["n"],
            "train_cls_loss": sums["cls"] / sums["n"],
            "train_sparsity_loss": sums["sparse"] / sums["n"],
            "test_accuracy": test_acc,
            "global_sparsity_pct": stats["global_sparsity_pct"],
            "per_layer_sparsity_pct":
                [l["sparsity_pct"] for l in stats["per_layer"]],
            "epoch_time_sec": time.time() - t0,
        })
        print(f"  [epoch {epoch:02d}] "
              f"lam_eff={current_lambda:.3f}  "
              f"cls={sums['cls']/sums['n']:.4f}  "
              f"test_acc={test_acc:5.2f}%  "
              f"sparsity={stats['global_sparsity_pct']:5.2f}%  "
              f"per_layer={[round(s,1) for s in history[-1]['per_layer_sparsity_pct']]}  "
              f"({time.time() - t0:.1f}s)")

    final_stats = model.sparsity_stats()
    final_acc = evaluate(model, test_loader, device)
    result = RunResult(
        lambda_val=lambda_val, epochs=epochs,
        final_train_loss=history[-1]["train_total_loss"],
        final_cls_loss=history[-1]["train_cls_loss"],
        final_sparsity_loss=history[-1]["train_sparsity_loss"],
        test_accuracy=final_acc,
        global_sparsity_pct=final_stats["global_sparsity_pct"],
        per_layer_sparsity=final_stats["per_layer"],
        history=history,
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = f"lambda_{lambda_val:g}".replace(".", "p")
        ckpt_path = save_dir / f"{tag}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "lambda_val": lambda_val,
            "result": asdict(result),
            "gate_values": model.all_gate_values(),
        }, ckpt_path)
        print(f"  saved checkpoint -> {ckpt_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.05, 0.25, 0.45])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sparsity-form", choices=["relu_shift", "sigmoid"],
                        default="relu_shift")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for lam in args.lambdas:
        result = train_one_run_synth(
            lambda_val=lam,
            epochs=args.epochs,
            lr=args.lr,
            gate_lr=args.gate_lr,
            batch_size=args.batch_size,
            warmup_epochs=args.warmup,
            sparsity_form=args.sparsity_form,
            device=device,
            save_dir=ckpt_dir,
        )
        all_results.append(asdict(result))
        with open(results_dir / "sweep_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n\nSUMMARY")
    print(f"{'Lambda':>10}  {'Test Acc':>10}  {'Sparsity':>10}")
    print("-" * 36)
    for r in all_results:
        print(f"{r['lambda_val']:>10.3f}  "
              f"{r['test_accuracy']:>9.2f}%  "
              f"{r['global_sparsity_pct']:>9.2f}%")


if __name__ == "__main__":
    main()
