"""
Evaluation / visualization for the self-pruning sweep.

Loads each checkpoint in ./checkpoints, plots:
  1. Per-λ gate distribution histogram (the key figure — shows the
     bimodal "pruned vs active" structure when pruning is working).
  2. Training trajectories (accuracy + sparsity over epochs) for all λ.
  3. Sparsity-vs-accuracy trade-off scatter.

Also writes a Markdown-friendly summary table of final metrics.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_checkpoints(ckpt_dir: Path) -> list[dict]:
    ckpts = []
    for p in sorted(ckpt_dir.glob("lambda_*.pt")):
        ck = torch.load(p, weights_only=False, map_location="cpu")
        ck["_path"] = p
        ckpts.append(ck)
    # Sort by lambda for consistent plotting order
    ckpts.sort(key=lambda c: c["lambda_val"])
    return ckpts


# ---------------------------------------------------------------------- #
# Plot 1: gate distribution histogram (the headline figure)
# ---------------------------------------------------------------------- #
def plot_gate_distribution(ckpts, out_path: Path):
    fig, axes = plt.subplots(1, len(ckpts), figsize=(5 * len(ckpts), 4),
                             sharey=True)
    if len(ckpts) == 1:
        axes = [axes]
    for ax, ck in zip(axes, ckpts):
        gates = ck["gate_values"].numpy()
        # Use log-y to see both the spike near 0 and the tail of active gates
        ax.hist(gates, bins=100, range=(0, 1),
                color="#2b7dd8", edgecolor="#0f3d7a",
                linewidth=0.3, alpha=0.9)
        ax.set_yscale("log")
        ax.axvline(0.01, color="#d84a2b", linestyle="--", linewidth=1.2,
                   label="prune threshold (0.01)")
        ax.set_xlabel("gate value  (sigmoid(score))")
        ax.set_title(
            f"λ = {ck['lambda_val']}   "
            f"acc={ck['result']['test_accuracy']:.1f}%   "
            f"sparsity={ck['result']['global_sparsity_pct']:.1f}%"
        )
        ax.legend(loc="upper center", fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("count (log scale)")
    fig.suptitle("Final gate-value distribution per λ", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------- #
# Plot 2: training trajectories
# ---------------------------------------------------------------------- #
def plot_trajectories(ckpts, out_path: Path):
    fig, (ax_acc, ax_sp) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ckpts)))
    for ck, color in zip(ckpts, colors):
        hist = ck["result"]["history"]
        epochs = [h["epoch"] for h in hist]
        acc = [h["test_accuracy"] for h in hist]
        sp = [h["global_sparsity_pct"] for h in hist]
        label = f"λ = {ck['lambda_val']}"
        ax_acc.plot(epochs, acc, marker="o", markersize=4,
                    linewidth=1.7, color=color, label=label)
        ax_sp.plot(epochs, sp, marker="s", markersize=4,
                   linewidth=1.7, color=color, label=label)

    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title("Test accuracy over training")
    ax_acc.grid(True, alpha=0.3); ax_acc.legend()

    ax_sp.set_xlabel("epoch"); ax_sp.set_ylabel("global sparsity (%)")
    ax_sp.set_title("Pruning progress over training")
    ax_sp.grid(True, alpha=0.3); ax_sp.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------- #
# Plot 3: sparsity-vs-accuracy trade-off
# ---------------------------------------------------------------------- #
def plot_tradeoff(ckpts, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    lams = [ck["lambda_val"] for ck in ckpts]
    accs = [ck["result"]["test_accuracy"] for ck in ckpts]
    sps = [ck["result"]["global_sparsity_pct"] for ck in ckpts]

    sc = ax.scatter(sps, accs, s=180,
                    c=np.log10(lams), cmap="plasma",
                    edgecolor="black", linewidth=1.0)
    for lam, sp, acc in zip(lams, sps, accs):
        ax.annotate(f"λ={lam:g}", (sp, acc),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10)

    ax.set_xlabel("Global sparsity (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs sparsity trade-off")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(λ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------- #
# Markdown summary
# ---------------------------------------------------------------------- #
def write_summary_table(ckpts, out_path: Path):
    lines = []
    lines.append("| λ (lambda) | Test Accuracy | Global Sparsity | Layer 0 | Layer 1 | Layer 2 | Layer 3 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for ck in ckpts:
        r = ck["result"]
        per = [l["sparsity_pct"] for l in r["per_layer_sparsity"]]
        lines.append(
            f"| {ck['lambda_val']} "
            f"| {r['test_accuracy']:.2f}% "
            f"| {r['global_sparsity_pct']:.2f}% "
            f"| {per[0]:.1f}% | {per[1]:.1f}% | {per[2]:.1f}% | {per[3]:.1f}% |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def main():
    ckpt_dir = Path("./checkpoints")
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    ckpts = load_checkpoints(ckpt_dir)
    if not ckpts:
        print("No checkpoints found. Run `python -m src.train_demo` or "
              "`python -m src.train` first.")
        return

    print(f"Loaded {len(ckpts)} checkpoint(s):")
    for ck in ckpts:
        r = ck["result"]
        print(f"  λ={ck['lambda_val']}  "
              f"acc={r['test_accuracy']:.2f}%  "
              f"sparsity={r['global_sparsity_pct']:.2f}%")

    print("\nGenerating plots...")
    plot_gate_distribution(ckpts, results_dir / "gate_distribution.png")
    plot_trajectories(ckpts, results_dir / "training_trajectories.png")
    plot_tradeoff(ckpts, results_dir / "tradeoff.png")
    write_summary_table(ckpts, results_dir / "summary_table.md")

    # Also write a consolidated JSON (handy for the API/frontend)
    payload = []
    for ck in ckpts:
        g = ck["gate_values"].numpy()
        # Bin the gate values into 40 buckets for the frontend histogram
        hist, edges = np.histogram(g, bins=40, range=(0, 1))
        payload.append({
            "lambda": ck["lambda_val"],
            "test_accuracy": ck["result"]["test_accuracy"],
            "global_sparsity_pct": ck["result"]["global_sparsity_pct"],
            "per_layer_sparsity": ck["result"]["per_layer_sparsity"],
            "history": ck["result"]["history"],
            "gate_histogram": {
                "bin_edges": edges.tolist(),
                "counts": hist.tolist(),
                "total_gates": int(g.size),
            },
        })
    with (results_dir / "dashboard.json").open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved {results_dir / 'dashboard.json'}")


if __name__ == "__main__":
    main()
