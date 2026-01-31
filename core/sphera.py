# core/sphera.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
import streamlit as st

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


# ============================ Utils internas ============================

def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n

def _l2_normalize_mat(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / norms

def _cosine_query_vs_matrix(q: np.ndarray, E: np.ndarray) -> np.ndarray:
    if q.ndim != 1:
        q = q.reshape(-1)
    qn = _l2_normalize_vec(q)
    En = _l2_normalize_mat(E)
    return En @ qn


# ========================= API usada no app =============================

def get_sphera_location_col(df: pd.DataFrame | None) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]:
        if c in df.columns:
            return c
    return None


def filter_sphera(df: pd.DataFrame | None, locations: List[str] | None, substr: str | None, years: int | None) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    # (filtros por years/substring iguais aos seus atuais — manter)
    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
        out = out[out[loc_col].astype(str).isin(set(locations))]
    if substr and "Description" in out.columns:
        out = out[out["Description"].astype(str).str.contains(substr, case=False, na=False)]
    return out if not out.empty else df

def topk_similar(
    query_text: str,
    df: pd.DataFrame | None,
    E: np.ndarray | None,
    topk: int = 20,
    min_sim: float = 0.30,
) -> List[Tuple[str, float, pd.Series]]:
    if not query_text or df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if E is None or getattr(E, "size", 0) == 0:
        return []

    model_name = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    enc = ensure_st_encoder(model_name)                     # <- retorna o encoder
    qv = encode_query(query_text, enc).astype(np.float32)   # <- ORDEM CORRETA!
    qv /= (np.linalg.norm(qv) + 1e-12)

    n = min(len(df), E.shape[0])
    if n == 0:
        return []
    E_view = E[:n, :]

    sims = (E_view @ qv).astype(float)
    order = np.argsort(-sims)

    id_col = None
    for cand in ("Event ID", "EVENT_ID", "EVENTID", "id", "ID"):
        if cand in df.columns:
            id_col = cand
            break

    out = []
    k = max(1, int(topk))
    for i in order[:k]:
        s = float(sims[i])
        if s < float(min_sim):
            continue
        row = df.iloc[i]
        evid = str(row.get(id_col, str(i)))
        out.append((evid, s, row))
    return out
