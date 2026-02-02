# core/sphera.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------
# Nome do modelo de embeddings a partir do config (com fallbacks)
# ---------------------------------------------------------------------
try:
    from config import OLLAMA_EMBEDDING_MODEL as EMBED_MODEL_NAME
except Exception:
    try:
        from config import EMBEDDING_MODEL as EMBED_MODEL_NAME
    except Exception:
        EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

from core.encoding import ensure_st_encoder, encode_query


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def get_sphera_location_col(df: pd.DataFrame | None) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]:
        if c in df.columns:
            return c
    return None


def filter_sphera(
    df: pd.DataFrame | None,
    locations: List[str] | None,
    substr: str | None,
    years: int | None
) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
