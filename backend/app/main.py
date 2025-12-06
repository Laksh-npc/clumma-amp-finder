from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import logging
import sys

# Optional torch import (model may be PyTorch-based)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Load .env from backend directory so env flags persist across reloads
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clumma")

app = FastAPI(title="CLuMMA AMP Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths and lazy loading
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
KERAS_MODEL_PATH = MODELS_DIR / "Model_weight_0.h5"
PRED_AHCP_DIR = MODELS_DIR / "pred-ahcp"
PRED_AHCP_WEIGHTS = PRED_AHCP_DIR / "Models" / "Model_weight_0.h5"

model = None            # Keras model
torch_model = None      # PyTorch model

def load_model_lazy():
    global model
    if model is not None:
        return model
    try:
        if not KERAS_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {KERAS_MODEL_PATH}")
        logger.info(f"Loading Keras model from {KERAS_MODEL_PATH}")
        m = load_model(KERAS_MODEL_PATH, compile=False)
        logger.info("Keras model loaded successfully")
        model = m
        return model
    except Exception as e:
        # Defer raising a descriptive error to the endpoint
        logger.error(f"Failed to load Keras model from {KERAS_MODEL_PATH}: {e}")
        raise RuntimeError(f"Failed to load Keras model from '{KERAS_MODEL_PATH}': {e}")

def load_torch_model_lazy():
    global torch_model
    if torch_model is not None:
        return torch_model
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed; cannot load pred-ahcp model")
    # Add pred-ahcp to import path to import layers.SNNModel
    sys.path.insert(0, str(PRED_AHCP_DIR))
    try:
        from layers import SNNModel  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import SNNModel from {PRED_AHCP_DIR}: {e}")
    if not PRED_AHCP_WEIGHTS.exists():
        raise FileNotFoundError(f"Torch weights not found at {PRED_AHCP_WEIGHTS}")
    try:
        logger.info(f"Loading PyTorch model from {PRED_AHCP_WEIGHTS}")
        # Build model with expected hyperparams from CLuMMA.py (64 channels, 2 heads, 32 out channels)
        dummy_shape = (1, 50, 26)
        m = SNNModel(dummy_shape, 64, 2, 32)  # type: ignore
        state = torch.load(PRED_AHCP_WEIGHTS, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        torch_model = m
        logger.info("PyTorch model loaded successfully")
        return torch_model
    except Exception as e:
        logger.error(f"Failed to load PyTorch model from {PRED_AHCP_WEIGHTS}: {e}")
        raise RuntimeError(f"Failed to load PyTorch model from '{PRED_AHCP_WEIGHTS}': {e}")

VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(VALID_AMINO_ACIDS)}
MAX_SEQUENCE_LENGTH = 50


def encode_sequence(seq: str, max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    seq = (seq or "").upper()
    seq = seq[:max_len]
    arr = np.zeros((max_len, len(VALID_AMINO_ACIDS)), dtype=np.float32)
    for t, ch in enumerate(seq):
        if ch in AA_INDEX:
            arr[t, AA_INDEX[ch]] = 1.0
    return arr

def encode_sequence_26_features(seq: str, max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    # Replicates CLuMMA.one_hot_padding features: 20 one-hot + 6 property flags
    seq = (seq or "").upper()
    seq = seq[:max_len]
    aa_list = list(VALID_AMINO_ACIDS)

    pos = set(['K', 'R'])
    neg = set(['D', 'E'])
    neut_charge = set(['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    hpho = set(['F', 'I', 'W', 'C'])
    pol = set(['K', 'D', 'E', 'Q', 'P', 'S', 'R', 'N', 'T', 'G'])
    neut_hydro = set(['A', 'H', 'Y', 'M', 'L', 'V'])

    feature_len = 26
    arr = np.zeros((max_len, feature_len), dtype=np.float32)
    for t, ch in enumerate(seq):
        if ch in aa_list:
            idx = aa_list.index(ch)
            arr[t, idx] = 1.0
            # Charge flags
            if ch in pos:
                arr[t, 20] = 1.0
            elif ch in neg:
                arr[t, 21] = 1.0
            elif ch in neut_charge:
                arr[t, 22] = 1.0
            # Hydrophobicity flags
            if ch in hpho:
                arr[t, 23] = 1.0
            elif ch in pol:
                arr[t, 24] = 1.0
            elif ch in neut_hydro:
                arr[t, 25] = 1.0
    return arr

class PredictItem(BaseModel):
    id: str
    sequence: str


class PredictRequest(BaseModel):
    items: List[PredictItem]


class PredictResponseItem(BaseModel):
    id: str
    sequence: str
    probability: float
    prediction: str  # "AMP" or "Non-AMP"
    confidence: str  # "High" | "Medium" | "Low"


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


def confidence_from_prob(p: float) -> str:
    if p > 0.8 or p < 0.2:
        return "High"
    if 0.3 <= p <= 0.7:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    keras_exists = KERAS_MODEL_PATH.exists()
    keras_size = KERAS_MODEL_PATH.stat().st_size if keras_exists else 0
    torch_exists = PRED_AHCP_WEIGHTS.exists()
    torch_size = PRED_AHCP_WEIGHTS.stat().st_size if torch_exists else 0
    logger.info(f"/health checked (keras_exists={keras_exists}, keras_size={keras_size}, torch_exists={torch_exists}, torch_size={torch_size})")
    return {
        "status": "ok",
        "keras_model_path": str(KERAS_MODEL_PATH),
        "keras_model_exists": keras_exists,
        "keras_model_size_bytes": keras_size,
        "torch_model_path": str(PRED_AHCP_WEIGHTS),
        "torch_model_exists": torch_exists,
        "torch_model_size_bytes": torch_size,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.items:
        return {"results": []}

    # Prefer PyTorch model if available
    use_torch = False
    if TORCH_AVAILABLE and PRED_AHCP_WEIGHTS.exists():
        try:
            tm = load_torch_model_lazy()
            use_torch = True
        except Exception as e:
            logger.warning(f"Torch model unavailable: {e}")

    if use_torch:
        # Encode sequences to 26-feature tensors
        logger.info("predict: using REAL MODEL (PyTorch) for inference")
        X_np = np.stack([encode_sequence_26_features(item.sequence) for item in req.items], axis=0)
        with torch.no_grad():  # type: ignore
            X_t = torch.from_numpy(X_np).float()  # type: ignore
            probs_t = tm(X_t)  # type: ignore
            probs_np = probs_t.squeeze().detach().cpu().numpy()
        if probs_np.ndim == 0:
            probs_np = np.array([float(probs_np)])
        results = []
        for item, p in zip(req.items, probs_np.tolist()):
            pred = "AMP" if p >= 0.5 else "Non-AMP"
            results.append(
                {
                    "id": item.id,
                    "sequence": item.sequence[:MAX_SEQUENCE_LENGTH],
                    "probability": float(p),
                    "prediction": pred,
                    "confidence": confidence_from_prob(float(p)),
                }
            )
        return {"results": results}

    # Else try Keras model
    try:
        m = load_model_lazy()
    except Exception as e:
        # If mock is allowed, return deterministic pseudo predictions to keep UI working
        allow_mock = os.environ.get("BACKEND_ALLOW_MOCK", "0") in ("1", "true", "True")
        if not allow_mock:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        # Mock path
        logger.warning("predict: using MOCK predictions (model not available)")
        def mock_prob(seq: str) -> float:
            # Simple stable hash-based mock probability on sequence content
            h = 0
            for ch in (seq or ""):
                h = (h * 131 + ord(ch)) & 0xFFFFFFFF
            return (h % 1000) / 1000.0
        results = []
        for item in req.items:
            p = mock_prob(item.sequence[:MAX_SEQUENCE_LENGTH])
            pred = "AMP" if p >= 0.5 else "Non-AMP"
            results.append(
                {
                    "id": item.id,
                    "sequence": item.sequence[:MAX_SEQUENCE_LENGTH],
                    "probability": float(p),
                    "prediction": pred,
                    "confidence": confidence_from_prob(float(p)),
                }
            )
        return {"results": results}

    X = np.stack([encode_sequence(item.sequence) for item in req.items], axis=0)
    if len(m.inputs[0].shape) == 4:
        X_model = np.expand_dims(X, axis=-1)
    else:
        X_model = X

    logger.info("predict: using REAL MODEL for inference")
    probs = m.predict(X_model, verbose=0).squeeze()
    if probs.ndim > 1:
        probs = probs[:, 0]

    results = []
    for item, p in zip(req.items, probs.tolist()):
        pred = "AMP" if p >= 0.5 else "Non-AMP"
        results.append(
            {
                "id": item.id,
                "sequence": item.sequence[:MAX_SEQUENCE_LENGTH],
                "probability": float(p),
                "prediction": pred,
                "confidence": confidence_from_prob(float(p)),
            }
        )
    return {"results": results}


