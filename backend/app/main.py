from pathlib import Path
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
import torch

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

PRED_AHCP_DIR = Path(__file__).resolve().parents[1] / "Pred-AHCP"
PRED_AHCP_WEIGHTS = PRED_AHCP_DIR / "models" / "Model_weight_4.h5"
VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_SEQUENCE_LENGTH = 50

model = None
def load_model_lazy():
    """Load and cache the SNNModel from Pred-AHCP"""
    global model
    if model is not None:
        return model
    
    sys.path.insert(0, str(PRED_AHCP_DIR))
    
    if not PRED_AHCP_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found at {PRED_AHCP_WEIGHTS}")
    
    try:
        logger.info(f"Loading SNNModel from {PRED_AHCP_WEIGHTS}")
        from layers import SNNModel
        
        input_shape = (1, MAX_SEQUENCE_LENGTH, 26)
        m = SNNModel(input_shape, in_channels=64, no_of_head=2, out_chan=32)
        
        checkpoint = torch.load(PRED_AHCP_WEIGHTS, map_location="cpu")
        m.load_state_dict(checkpoint)
        m.eval()
        
        model = m
        logger.info("SNNModel loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load SNNModel: {e}")
        raise RuntimeError(f"Model loading failed: {e}")



def encode_sequence_26_features(seq: str, max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    """Encode amino acid sequence with 26 features: 20 one-hot + 6 physicochemical properties"""
    seq = (seq or "").upper()[:max_len]
    aa_list = list(VALID_AMINO_ACIDS)
    
    pos_charged = {'K', 'R'}
    neg_charged = {'D', 'E'}
    neut_charge = {'A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'}
    hydrophobic = {'F', 'I', 'W', 'C'}
    polar = {'K', 'D', 'E', 'Q', 'P', 'S', 'R', 'N', 'T', 'G'}
    neut_hydro = {'A', 'H', 'Y', 'M', 'L', 'V'}
    
    arr = np.zeros((max_len, 26), dtype=np.float32)
    
    for t, ch in enumerate(seq):
        if ch in aa_list:
            arr[t, aa_list.index(ch)] = 1.0
            
            if ch in pos_charged:
                arr[t, 20] = 1.0
            elif ch in neg_charged:
                arr[t, 21] = 1.0
            elif ch in neut_charge:
                arr[t, 22] = 1.0
            
            if ch in hydrophobic:
                arr[t, 23] = 1.0
            elif ch in polar:
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
    """
    Calculate confidence based on probability.
    High confidence for extreme probabilities (close to 0 or 1)
    Low confidence for probabilities near 0.5 (uncertain)
    """
    if p >= 0.8 or p <= 0.2:
        return "High"
    if (0.65 <= p < 0.8) or (0.2 < p <= 0.35):
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    model_exists = PRED_AHCP_WEIGHTS.exists()
    model_size = PRED_AHCP_WEIGHTS.stat().st_size if model_exists else 0
    logger.info(f"/health checked (model_exists={model_exists}, model_size={model_size})")
    return {
        "status": "ok",
        "model_path": str(PRED_AHCP_WEIGHTS),
        "model_exists": model_exists,
        "model_size_bytes": model_size,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict AMP (Antimicrobial Peptide) probability for amino acid sequences
    
    Uses SNNModel from Pred-AHCP with 26-feature encoding:
    - 20 features: one-hot encoding for amino acids (ACDEFGHIKLMNPQRSTVWY)
    - 6 features: physicochemical properties (charge × 3, hydrophobicity × 3)
    
    Returns prediction ("AMP" if probability >= 0.5), raw probability, and confidence
    """
    if not req.items:
        return {"results": []}
    try:
        m = load_model_lazy()
        logger.info("Using SNNModel from Pred-AHCP for inference")
        
        X_np = np.stack([encode_sequence_26_features(item.sequence) for item in req.items], axis=0)
        
        with torch.no_grad():
            X_t = torch.from_numpy(X_np).float()
            probs_t = m(X_t)
            print(probs_t)
            probs_np = probs_t.detach().cpu().numpy()
        
        if probs_np.ndim > 1:
            probs_np = probs_np.flatten()
        if probs_np.ndim == 0:
            probs_np = np.array([float(probs_np)])
        
        results = []
        for item, p in zip(req.items, probs_np):
            p_float = float(p)
            pred = "AMP" if p_float >= 0.5 else "Non-AMP"
            results.append({
                "id": item.id,
                "sequence": item.sequence[:MAX_SEQUENCE_LENGTH],
                "probability": p_float,
                "prediction": pred,
                "confidence": confidence_from_prob(p_float),
            })
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


