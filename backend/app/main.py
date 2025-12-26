from pathlib import Path
from datetime import datetime
import uuid
import shutil
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

PMODEL_NEW_DIR = Path(__file__).resolve().parents[1] / "Pmodel_new"
PMODEL_NEW_WEIGHTS = PMODEL_NEW_DIR / "Models" / "Model_weight_4.h5"
ATTENTION_BASE_DIR = Path(__file__).resolve().parents[1] / "attention"
VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_SEQUENCE_LENGTH = 50


def one_hot_padding(seq_list, padding):
    """Generate features for aa sequences [one-hot encoding with zero padding].
    35 features: 20 one-hot + 15 physicochemical properties
    EXACT copy from Pmodel_new/test.py
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    pos, neg, neut_charge = ['K', 'R'], ['D', 'E'], ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    hpho, pol, neut_hydro = ['F', 'I', 'W', 'C'], ['K', 'D', 'E', 'Q', 'P', 'S', 'R', 'N', 'T', 'G'], ['A', 'H', 'Y', 'M', 'L', 'V']
    g1_p, g2_p, g3_p = ['G', 'A', 'S', 'D', 'T'], ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'], ['K', 'M', 'H', 'F', 'R', 'Y', 'W']
    g1_v, g2_v, g3_v = ['G', 'A', 'S', 'T', 'P', 'D'], ['N', 'V', 'E', 'Q', 'I', 'L'], ['M', 'H', 'K', 'F', 'R', 'Y', 'W']
    H, S, C = ['E','A', 'L', 'M', 'Q', 'K', 'R', 'H'], ['V','I','Y','C','W','F','T'], ['G', 'N', 'P', 'S', 'D']

    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*35
        one_hot[aa[i]][i] = 1 
        if aa[i] in pos:
            one_hot[aa[i]][20]=1
        elif aa[i] in neg:
            one_hot[aa[i]][21]=1
        elif aa[i] in neut_charge:
            one_hot[aa[i]][22]=1
        
        if aa[i] in hpho:
            one_hot[aa[i]][23]=1
        elif aa[i] in pol:
            one_hot[aa[i]][24]=1
        elif aa[i] in neut_hydro:
            one_hot[aa[i]][25]=1

        if aa[i] in H:
            one_hot[aa[i]][26]=1
        elif aa[i] in S:
            one_hot[aa[i]][27]=1
        elif aa[i] in C:
            one_hot[aa[i]][28]=1

        if aa[i] in g1_p:
            one_hot[aa[i]][29]=1
        elif aa[i] in g2_p:
            one_hot[aa[i]][30]=1
        elif aa[i] in g3_p:
            one_hot[aa[i]][31]=1

        if aa[i] in g1_v:
            one_hot[aa[i]][32]=1
        elif aa[i] in g2_v:
            one_hot[aa[i]][33]=1
        elif aa[i] in g3_v:
            one_hot[aa[i]][34]=1
    
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*35]*(padding-len(seq_list[i]))
        feat_list.append(feat)   
        
    feat_list = torch.from_numpy(np.array(feat_list))

    return feat_list


def validate_sequence(seq: str) -> str:
    seq = seq.upper()
    invalid = set(seq) - set(VALID_AMINO_ACIDS)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid amino acids found: {''.join(sorted(invalid))}. "
                f"Allowed amino acids: {VALID_AMINO_ACIDS}"
            )
        )
    return seq


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
    attention_dir: str


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
    model_exists = PMODEL_NEW_WEIGHTS.exists()
    model_size = PMODEL_NEW_WEIGHTS.stat().st_size if model_exists else 0
    logger.info(f"/health checked (model_exists={model_exists}, model_size={model_size})")
    return {
        "status": "ok",
        "model_path": str(PMODEL_NEW_WEIGHTS),
        "model_exists": model_exists,
        "model_size_bytes": model_size,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict AMP (Antimicrobial Peptide) probability for amino acid sequences
    
    Uses SNNModel from Pmodel_new with 35-feature encoding:
    - 20 features: one-hot encoding for amino acids (ACDEFGHIKLMNPQRSTVWY)
    - 15 features: physicochemical properties (charge × 3, hydrophobicity × 3, 
                   secondary structure × 3, polarity groups × 6)
    
    Returns prediction ("AMP" if probability > 0.5), raw probability, and confidence
    """
    if not req.items:
        return {"results": []}
    try:
        # Import SNNModel from Pmodel_new - EXACT same as test.py
        sys.path.insert(0, str(PMODEL_NEW_DIR))
        from layers import SNNModel
        
        # Prepare sequences for encoding - EXACT same as test.py
        seq_list = [
            validate_sequence(item.sequence)[:MAX_SEQUENCE_LENGTH]
            for item in req.items
        ]
        X_test = one_hot_padding(seq_list, MAX_SEQUENCE_LENGTH)
        
        # Load model with ACTUAL batch shape like test.py does
        # test.py: model = SNNModel(X_test.shape, in_channels, heads, out_channels)
        in_channels = 64
        heads = 2
        out_channels = 32
        
        m = SNNModel(X_test.shape, in_channels, heads, out_channels)
        m.load_state_dict(torch.load(PMODEL_NEW_WEIGHTS, map_location="cpu"))
        
        # EXACT inference logic from test.py (no eval(), no no_grad())
        # Create per-request attention output directory
        try:
            ATTENTION_BASE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        req_stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        req_dir = ATTENTION_BASE_DIR / f"req-{req_stamp}-{uuid.uuid4().hex[:8]}"
        req_dir.mkdir(parents=True, exist_ok=True)

        # Run model forward (this writes Attention_*.csv in CWD via layers.py)
        probs_t = torch.squeeze(m(X_test.float()))
        probs_np = probs_t.detach().numpy()

        # Move any Attention_*.csv created by the model to the per-request folder
        try:
            for f in Path.cwd().glob("Attention_*.csv"):
                shutil.move(str(f), str(req_dir / f.name))
        except Exception:
            # Non-fatal: if attention CSVs are not found or cannot be moved, continue
            pass
        
        results = []
        for item, p in zip(req.items, probs_np):
            p_float = float(p)
            pred = "AMP" if p_float > 0.5 else "Non-AMP"
            results.append({
                "id": item.id,
                "sequence": item.sequence[:MAX_SEQUENCE_LENGTH],
                "probability": p_float,
                "prediction": pred,
                "confidence": confidence_from_prob(p_float),
            })
        return {"results": results, "attention_dir": str(req_dir)}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


