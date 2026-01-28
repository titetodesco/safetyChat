
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import json

from ..config import (
    SPH_PQ_PATH, SPH_NPZ_PATH,
    PROMPTS_MD_PATH, DATASETS_CONTEXT_PATH,
    WS_NPZ, WS_LBL, PREC_NPZ, PREC_LBL, CP_NPZ_MAIN, CP_NPZ_ALT, CP_LBL_PARQ, CP_LBL_JSONL
)

@st.cache_data(show_spinner=False)
def load_npz_embeddings(path: Path):
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for key in ("embeddings","E","X","vectors","vecs","arr_0"):
                if key in z:
                    E = z[key].astype(np.float32, copy=False)
                    # normalize
                    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                    return (E / n).astype(np.float32)
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def load_parquet_safe(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_prompts_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def load_datasets_context(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def load_sphera():
    df = load_parquet_safe(SPH_PQ_PATH)
    E = load_npz_embeddings(SPH_NPZ_PATH)
    return df, E

@st.cache_data(show_spinner=False)
def _load_npz_any(path: Path):
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for k in ("embeddings","E","X","vectors","vecs","arr_0"):
                if k in z:
                    V = z[k].astype(np.float32, copy=False)
                    n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
                    return (V / n).astype(np.float32)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def _load_labels_any(parq: Path, jsonl: Path):
    import pandas as pd, json
    if parq.exists():
        try:
            df = pd.read_parquet(parq)
            return df
        except Exception:
            pass
    if jsonl.exists():
        try:
            rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
            return pd.DataFrame(rows)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
def load_dicts():
    E_ws   = load_npz_embeddings(WS_NPZ)
    L_ws   = load_parquet_safe(WS_LBL)
    E_prec = load_npz_embeddings(PREC_NPZ)
    L_prec = load_parquet_safe(PREC_LBL)
    E_cp   = _load_npz_any(CP_NPZ_MAIN) or _load_npz_any(CP_NPZ_ALT)
    L_cp   = _load_labels_any(CP_LBL_PARQ, CP_LBL_JSONL)
    return (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp)
