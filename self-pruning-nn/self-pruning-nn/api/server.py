"""
FastAPI backend for the self-pruning NN dashboard.

Endpoints:
    GET  /               - health check
    GET  /api/sweep      - full sweep results (all lambdas, for dashboard)
    GET  /api/summary    - compact summary table (for quick display)
    GET  /api/gates/{lambda_tag} - gate histogram + per-layer stats for one run
    POST /api/predict    - run a forward pass on a submitted 32x32x3 image
                           (base64 PNG/JPEG) and return class probs
                           plus which fraction of gates were active

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The model lives in src/, which we reach via the repo root.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import SelfPruningNet  # noqa: E402

app = FastAPI(title="Self-Pruning NN Dashboard API", version="1.0.0")

# Wide-open CORS — it's a local dashboard, not a public API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- #
# State: load all checkpoints into memory at startup
# ---------------------------------------------------------------------- #
class AppState:
    sweep_payload: list[dict] = []
    models: dict[str, SelfPruningNet] = {}      # tag -> loaded model
    best_tag: Optional[str] = None              # best accuracy/sparsity combo


STATE = AppState()

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Synthetic dataset normalization (matches src/synth_data.py)
SYNTH_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
SYNTH_STD = np.array([0.25, 0.25, 0.25], dtype=np.float32).reshape(3, 1, 1)


def _lambda_tag(lam: float) -> str:
    """Same tag format as train_demo.py uses for checkpoint filenames."""
    return f"lambda_{lam:g}".replace(".", "p")


def load_state() -> None:
    dashboard_json = ROOT / "results" / "dashboard.json"
    ckpt_dir = ROOT / "checkpoints"

    if not dashboard_json.exists():
        print(f"[warn] {dashboard_json} missing. Run `python -m src.evaluate` first.")
        return
    STATE.sweep_payload = json.loads(dashboard_json.read_text())

    # Load each checkpoint so /api/predict can use any of them
    for entry in STATE.sweep_payload:
        lam = entry["lambda"]
        tag = _lambda_tag(lam)
        ckpt_path = ckpt_dir / f"{tag}.pt"
        if not ckpt_path.exists():
            print(f"[warn] missing checkpoint: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model = SelfPruningNet()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        STATE.models[tag] = model

    # Pick the best trade-off: highest (acc * (1 + sparsity/100)) score
    # This rewards a model that is both accurate AND sparse.
    def score(entry):
        acc = entry["test_accuracy"] / 100.0
        sp = entry["global_sparsity_pct"] / 100.0
        return acc * (1.0 + sp)

    best = max(STATE.sweep_payload, key=score)
    STATE.best_tag = _lambda_tag(best["lambda"])
    print(f"[ok] loaded {len(STATE.models)} model(s); best = {STATE.best_tag}")


@app.on_event("startup")
def _startup():
    load_state()


# ---------------------------------------------------------------------- #
# Endpoints
# ---------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "service": "self-pruning-nn-api",
        "version": "1.0.0",
        "runs_loaded": len(STATE.sweep_payload),
        "models_loaded": list(STATE.models.keys()),
        "best_model_tag": STATE.best_tag,
    }


@app.get("/api/sweep")
def sweep():
    """Full sweep payload — used by the dashboard to render all charts."""
    if not STATE.sweep_payload:
        raise HTTPException(503, "No sweep results loaded. "
                                 "Run training + evaluate first.")
    return STATE.sweep_payload


@app.get("/api/summary")
def summary():
    """Compact summary for the dashboard header."""
    rows = []
    for entry in STATE.sweep_payload:
        rows.append({
            "lambda": entry["lambda"],
            "test_accuracy": entry["test_accuracy"],
            "global_sparsity_pct": entry["global_sparsity_pct"],
            "per_layer_sparsity_pct": [
                round(l["sparsity_pct"], 2) for l in entry["per_layer_sparsity"]
            ],
            "active_gates": entry["gate_histogram"]["total_gates"]
                            - int(round(
                                entry["gate_histogram"]["total_gates"]
                                * entry["global_sparsity_pct"] / 100.0
                            )),
            "total_gates": entry["gate_histogram"]["total_gates"],
        })
    return {"rows": rows, "best_tag": STATE.best_tag}


@app.get("/api/gates/{lambda_tag}")
def gates_for_run(lambda_tag: str):
    """Return the gate histogram and per-layer stats for a single run."""
    # Accept either "0p25" or "lambda_0p25"
    tag = lambda_tag if lambda_tag.startswith("lambda_") else f"lambda_{lambda_tag}"
    entry = next(
        (e for e in STATE.sweep_payload if _lambda_tag(e["lambda"]) == tag),
        None,
    )
    if entry is None:
        raise HTTPException(404, f"No run found for tag {tag}")
    return entry


# ---------------------------------------------------------------------- #
# /api/predict
# ---------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description=("Base64-encoded 32x32 image. Accepts raw base64 "
                     "or 'data:image/png;base64,...' prefix."),
    )
    lambda_tag: Optional[str] = Field(
        default=None,
        description="Checkpoint to use, e.g. '0p25'. Default: best model.",
    )


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    probs: list[float]
    lambda_used: float
    global_sparsity_pct: float


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not STATE.models:
        raise HTTPException(503, "No models loaded.")

    tag = req.lambda_tag or STATE.best_tag
    if tag and not tag.startswith("lambda_"):
        tag = f"lambda_{tag}"
    model = STATE.models.get(tag)
    if model is None:
        raise HTTPException(404, f"Model tag {tag} not found. "
                                 f"Available: {list(STATE.models.keys())}")

    # Decode the image
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(500, "Pillow not installed on the server.")

    data = req.image_base64
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((32, 32))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - SYNTH_MEAN) / SYNTH_STD
    x = torch.from_numpy(arr).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    predicted_index = int(np.argmax(probs))

    # Find the matching sweep entry for sparsity metadata
    lam = next(e["lambda"] for e in STATE.sweep_payload
               if _lambda_tag(e["lambda"]) == tag)
    sp = next(e["global_sparsity_pct"] for e in STATE.sweep_payload
              if _lambda_tag(e["lambda"]) == tag)

    return PredictResponse(
        predicted_class=CIFAR10_CLASSES[predicted_index],
        predicted_index=predicted_index,
        probs=probs,
        lambda_used=lam,
        global_sparsity_pct=sp,
    )
