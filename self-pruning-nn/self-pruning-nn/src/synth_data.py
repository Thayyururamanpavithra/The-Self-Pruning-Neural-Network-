"""
Build a synthetic CIFAR-10-shaped dataset for offline demonstration
when real CIFAR-10 download is unavailable.

Each class has a distinct low-frequency color/texture pattern so a
shallow MLP can actually learn it to ~60-70% accuracy, mimicking
what the real script would do on true CIFAR-10 (target ~50% for a
flat MLP baseline).
"""
from pathlib import Path
import numpy as np
import torch


def make_synthetic_cifar(n_train=10000, n_test=2000, seed=0, out_dir="./data/synth"):
    rng = np.random.default_rng(seed)
    H = W = 32
    num_classes = 10

    # Per-class "prototype": mean color + two sinusoidal texture axes.
    # Overlapping color ranges + strong noise keep this from being trivial.
    proto_means = rng.uniform(0.35, 0.65, size=(num_classes, 3)).astype(np.float32)
    proto_freqs = rng.uniform(0.5, 3.0, size=(num_classes, 2)).astype(np.float32)
    proto_phase = rng.uniform(0, 2 * np.pi, size=(num_classes,)).astype(np.float32)

    xs_grid, ys_grid = np.meshgrid(
        np.linspace(0, 1, W, dtype=np.float32),
        np.linspace(0, 1, H, dtype=np.float32),
        indexing="xy",
    )

    def gen(n):
        labels = rng.integers(0, num_classes, size=n)
        imgs = np.empty((n, 3, H, W), dtype=np.float32)
        for i, c in enumerate(labels):
            fx, fy = proto_freqs[c]
            phase = proto_phase[c]
            # Jittered frequencies per sample add within-class variation
            fx_j = fx + rng.normal(0, 0.3)
            fy_j = fy + rng.normal(0, 0.3)
            pattern = 0.5 + 0.25 * np.sin(
                2 * np.pi * fx_j * xs_grid + 2 * np.pi * fy_j * ys_grid + phase
            )
            for ch in range(3):
                color_jitter = rng.normal(0, 0.08)
                noise = rng.normal(0, 0.22, size=(H, W)).astype(np.float32)
                imgs[i, ch] = np.clip(
                    (proto_means[c, ch] + color_jitter) * pattern + noise,
                    0.0, 1.0,
                )
        return imgs, labels.astype(np.int64)

    x_train, y_train = gen(n_train)
    x_test, y_test = gen(n_test)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(out / "test.npz", x=x_test, y=y_test)
    print(f"Saved synthetic data → {out}")
    print(f"  train: x={x_train.shape}  y={y_train.shape}")
    print(f"  test:  x={x_test.shape}   y={y_test.shape}")


class SynthCIFAR(torch.utils.data.Dataset):
    """Loads the .npz files produced above and applies basic normalization."""
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.25, 0.25, 0.25], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.x = d["x"]
        self.y = d["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = (self.x[idx] - self.mean) / self.std
        return torch.from_numpy(x), int(self.y[idx])


if __name__ == "__main__":
    make_synthetic_cifar()
