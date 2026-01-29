
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import json

# import tolerante das constantes
try:
    from config import (
        # Sphera
        SPH_PQ_PATH, SPH_NPZ_PATH,
        # Dicionários WS/Precursores
        WS_NPZ, WS_LBL_PARQ, WS_LBL_JSONL,
        PREC_NPZ, PREC_LBL_PARQ, PREC_LBL_JSONL,
        # CP
        CP_NPZ_MAIN, CP_NPZ_ALT, CP_LBL_PARQ, CP_LBL_JSONL,
        # History/GoSee (se houver)
        GOSEE_PQ_PATH, GOSEE_NPZ_PATH,
        INC_PQ_PATH, INC_NPZ_PATH, INC_JSONL_PATH,
    )
except Exception:
    SPH_PQ_PATH = SPH_NPZ_PATH = None
    WS_NPZ = WS_LBL_PARQ = WS_LBL_JSONL = None
    PREC_NPZ = PREC_LBL_PARQ = PREC_LBL_JSONL = None
    CP_NPZ_MAIN = CP_NPZ_ALT = CP_LBL_PARQ = CP_LBL_JSONL = None
    GOSEE_PQ_PATH = GOSEE_NPZ_PATH = None
    INC_PQ_PATH = INC_NPZ_PATH = INC_JSONL_PATH = None

@st.cache_data(show_spinner=False)
def _load_parquet(path: Path|None):
    if not path or not Path(path).exists(): return None
    try: return pd.read_parquet(path)
    except Exception: return None

@st.cache_data(show_spinner=False)
def _load_jsonl(path: Path|None):
    if not path or not Path(path).exists(): return None
    try:
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        return pd.DataFrame(rows)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_npz_embeddings_any(path: Path|None):
    if not path or not Path(path).exists(): return None
    try:
        z = np.load(str(path), allow_pickle=True)
        for k in ("embeddings","E","X","vectors","vecs","arr_0"):
            if k in z:
                E = z[k].astype(np.float32, copy=False)
                n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                return (E / n).astype(np.float32)
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def _load_labels_any(parquet_path: Path|None, jsonl_path: Path|None):
    return _load_parquet(parquet_path) or _load_jsonl(jsonl_path)

@st.cache_data(show_spinner=False)
def load_dicts():
    # WS
    E_ws = _load_npz_embeddings_any(WS_NPZ)
    L_ws = _load_labels_any(WS_LBL_PARQ, WS_LBL_JSONL)
    # Precursores
    E_prec = _load_npz_embeddings_any(PREC_NPZ)
    L_prec = _load_labels_any(PREC_LBL_PARQ, PREC_LBL_JSONL)
    # CP
    E_cp = _load_npz_embeddings_any(CP_NPZ_MAIN) or _load_npz_embeddings_any(CP_NPZ_ALT)
    L_cp = _load_labels_any(CP_LBL_PARQ, CP_LBL_JSONL)
    return E_ws, L_ws, E_prec, L_prec, E_cp, L_cp
None
