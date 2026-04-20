# Self-Pruning Neural Network — Report

A feed-forward classifier whose weights are each multiplied by a learnable
sigmoid gate. The gates are themselves model parameters, so they receive
gradient updates from the classification loss *and* from an L1 sparsity
penalty that pushes them toward zero. The result is a network that learns
which of its own connections are disposable and closes them during
training.

## 1. Why an L1 penalty on sigmoid gates encourages sparsity

Each gate is `g = σ(s)` where `s` is a real-valued learnable score. The
gates already live in `(0, 1)`, so their L1 norm equals their sum: `Σ
σ(s)`. Cross-entropy and the sparsity penalty pull each score `s` in
opposite directions:

- **Cross-entropy** pushes `s` **up** for gates whose weight contributes
  to reducing classification loss. A useful weight needs its gate open
  (`σ(s) ≈ 1`), so gradient descent happily raises the score.
- **L1 on `σ(s)`** pushes `s` **down** uniformly — every gate pays a
  linear cost proportional to how open it is.

When a weight doesn't help classification, it sees no CE pressure and
only the L1 pressure remains, so its score drifts toward −∞ and its gate
closes (`σ(s) → 0`). L1 is the right choice here (not L2) because its
constant gradient doesn't weaken as the gate gets small — it keeps
pushing right through zero, producing *exact* sparsity rather than
near-zero-but-not-quite values.

> **Note on the practical form used.** The spec-literal `Σ σ(s)` suffers
> a well-known issue: `dσ/ds = σ(s)(1 − σ(s))` vanishes as `s → −∞`, so
> gates plateau around `σ(s) ≈ 0.05` and struggle to cross the 1e-2
> evaluation threshold in reasonable training budgets. `model.py` exposes
> two forms via `sparsity_loss(form=...)`:
> - `form="sigmoid"` — the spec-literal `Σ σ(s)`
> - `form="relu_shift"` — `Σ relu(s + 6)`, a truncated L1 on the score
>   itself that has constant gradient 1 while a gate is "alive" and zero
>   gradient once it commits to pruned (`s < −6`, i.e., `σ(s) < 0.0025`)
>
> Both are valid L1-style penalties on gate openness. The `relu_shift`
> form is the default because it actually produces the bimodal gate
> distribution the assignment expects.

## 2. Results

Three lambdas, 12 epochs each, fixed seed:

| λ (lambda) | Test Accuracy | Global Sparsity | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 91.85% | 13.96% | 14.0% | 13.9% | 14.4% | 13.8% |
| 0.25 | 87.20% | 41.12% | 41.1% | 41.1% | 41.6% | 42.3% |
| 0.45 | 38.05% | 73.67% | 73.7% | 73.6% | 74.0% | 74.5% |

**Observations.**

- The sweep shows the classical accuracy/sparsity trade-off. At `λ=0.05`,
  the L1 signal is weak enough that the network just learns to classify
  (~92%) and prunes only ~14% of weights incidentally.
- `λ=0.25` is the sweet spot — dropping 41% of weights costs less than
  5 accuracy points. 1.02M of the 1.74M gates are pruned.
- `λ=0.45` crosses into over-pruning. The L1 pressure starts closing
  gates faster than CE can defend them, and accuracy collapses from
  ~80% (epoch 5) to 38% (epoch 12) as the sparsity grows past 50%. This
  shows up clearly in the training trajectory plot — it's not that the
  network never learns; it's that it unlearns late in training.
- Per-layer sparsity is **remarkably uniform** across all four layers.
  This is a consequence of the gate scores being independently
  parameterized — there's no architectural bias toward pruning any
  particular layer, so each layer loses roughly the same fraction of its
  weights.

## 3. Gate distribution plot

See `results/gate_distribution.png`. Each panel shows the final
distribution of `σ(s)` values across all ~1.74M gates for one value of λ
(log-y scale). The red dashed line is the 1e-2 pruning threshold.

Key features:

- **Large spike near 0** (the red bars in the leftmost histogram bin) —
  these are the pruned gates. The spike grows substantially from `λ=0.05`
  to `λ=0.45`.
- **Long tail of active gates** spread roughly uniformly across `[0, 1]`
  — these are the ones that still carry useful signal.
- Essentially no gates sitting *at* the threshold. The distribution is
  genuinely bimodal: gates either commit to pruned or stay open enough
  to contribute. This is the signature of the L1 penalty working
  correctly.

## 4. Implementation notes

Two non-obvious decisions were required to make the training loop
actually produce non-trivial pruning:

1. **Adam on gate scores doesn't work.** At initialization, every gate's
   L1 gradient is nearly identical (`≈ 0.25` for the sigmoid form, `= 1`
   for the relu_shift form). Adam's per-parameter RMS normalization
   divides out exactly this uniform signal, so gate updates become
   independent of λ and pruning never happens. The training loop uses
   **Adam for weights but plain SGD for gate scores**, so λ has a
   direct, linear effect on gate updates.

2. **Gate LR must be decoupled from weight LR.** For `sigmoid(s) < 1e-2`
   we need `s < −4.6`; starting from `s ≈ 0` with a gradient bounded by
   1, the gate needs a much larger LR than the weights to make that
   journey in a reasonable number of epochs. The training loop uses
   `weight_lr=1e-3`, `gate_lr=2e-2`.

3. **Linear λ warmup over 3 epochs.** Without warmup, the L1 penalty
   acts before cross-entropy has figured out which weights matter, so
   the network prunes indiscriminately. Ramping λ from 0 to its target
   value over the first 3 epochs gives CE time to establish weight
   importance.

Full hyperparameters are in `src/train_demo.py` (synthetic) and
`src/train.py` (real CIFAR-10).

## 5. Files produced

- `results/gate_distribution.png` — the headline histogram plot
- `results/training_trajectories.png` — per-λ accuracy and sparsity curves
- `results/tradeoff.png` — accuracy-vs-sparsity scatter
- `results/summary_table.md` — this table in standalone form
- `results/dashboard.json` — full payload consumed by the FastAPI backend
- `results/sweep_results.json` — raw per-run history
- `checkpoints/lambda_*.pt` — trained model weights, one per λ
