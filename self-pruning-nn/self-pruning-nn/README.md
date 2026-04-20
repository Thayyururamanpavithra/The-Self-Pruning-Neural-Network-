# Self-Pruning Neural Network

A neural network that learns to prune its own connections during training.
Each weight is multiplied by a learnable sigmoid gate; an L1 penalty on
those gates drives most of them to exactly zero, leaving a sparse
sub-network of only the important connections.

Built end-to-end with:

- **PyTorch** — custom `PrunableLinear` layer + training loop
- **FastAPI** — serves sweep results + live inference endpoints
- **React + Recharts + Vite** — dark editorial dashboard for visualizing
  the accuracy/sparsity trade-off
- **Matplotlib** — static plots for the report

```
self-pruning-nn/
├── src/
│   ├── model.py          # PrunableLinear + SelfPruningNet
│   ├── train.py          # Real CIFAR-10 training (downloads via torchvision)
│   ├── train_demo.py     # Synthetic-data fallback (for offline runs)
│   ├── synth_data.py     # Builds a CIFAR-shaped synthetic dataset
│   └── evaluate.py       # Builds the plots + dashboard.json
├── api/
│   └── server.py         # FastAPI backend
├── frontend/             # React dashboard
│   ├── src/App.jsx
│   ├── src/index.css
│   └── ...
├── results/              # Generated plots, JSON, summary table
├── checkpoints/          # Trained model weights per lambda
├── report.md             # Write-up with analysis
├── requirements.txt
└── README.md
```

## Quick start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

**On real CIFAR-10 (recommended):**

```bash
python -m src.train --lambdas 0.05 0.25 0.45 --epochs 20
```

This downloads CIFAR-10 via torchvision and runs the 3-lambda sweep.
Takes ~15 minutes on a modern GPU, ~2 hours on CPU.

**Offline / no-internet fallback:** there's a synthetic CIFAR-like
dataset generator that reproduces the accuracy/sparsity trade-off behavior
for debugging without needing the real dataset:

```bash
python -m src.train_demo --lambdas 0.05 0.25 0.45 --epochs 12
```

Takes ~2 minutes on CPU.

### 3. Generate plots and the dashboard payload

```bash
python -m src.evaluate
```

This reads every checkpoint in `checkpoints/` and writes:

- `results/gate_distribution.png` — the headline bimodal histogram
- `results/training_trajectories.png`
- `results/tradeoff.png`
- `results/summary_table.md`
- `results/dashboard.json` — consumed by the backend

### 4. Run the dashboard

In one terminal:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

In another:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` calls
to the FastAPI backend on port 8000.

## What the dashboard shows

- **Summary cards** — one per λ, highlighting the optimal trade-off
- **Gate distribution histogram** — interactive, switch between λ values
  to watch the spike at zero grow
- **Training trajectories** — accuracy and sparsity over epochs, one
  line per λ
- **Per-layer sparsity table** — how many gates each layer lost
- **Live prediction panel** — upload a 32x32 image, get class
  probabilities from any of the pruned models

## API endpoints

| Route | Description |
|---|---|
| `GET /` | Health check |
| `GET /api/sweep` | Full per-λ payload (history + gate histogram) |
| `GET /api/summary` | Compact summary for dashboard header |
| `GET /api/gates/{lambda_tag}` | Single run details (e.g. `0p25` for λ=0.25) |
| `POST /api/predict` | Run inference on a base64-encoded 32x32 image |

## Reproducing the headline result

Default hyperparameters (`lr=1e-3`, `gate_lr=0.02`, `warmup=3`, 12 epochs,
batch 256) produce this trade-off on the synthetic benchmark:

| λ | Test Acc | Global Sparsity |
|---:|---:|---:|
| 0.05 | 91.85% | 13.96% |
| 0.25 | 87.20% | 41.12% |
| 0.45 | 38.05% | 73.67% |

See `report.md` for the full analysis of *why* these specific settings
produce non-trivial pruning (spoiler: Adam cancels the L1 gradient on
gate scores; gates need their own optimizer with a much higher LR).

## Troubleshooting

- **Dashboard shows "Could not reach API"** — make sure `uvicorn` is
  running on port 8000.
- **No checkpoints found** — run `python -m src.train_demo` (or
  `src.train`) first, then `python -m src.evaluate`.
- **Everything-zero sparsity** — you're probably running with the
  sigmoid form (`--sparsity-form sigmoid`), which plateaus. Use the
  default `relu_shift` form, or train for many more epochs.
- **Accuracy crashes at epoch 6-8** — λ is too high; try 0.1-0.2 range.
