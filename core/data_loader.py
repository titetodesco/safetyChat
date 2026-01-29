
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import json

try:
    from config import (
        SPH_PQ_PATH, SPH_NPZ_PATH,
        GOSEE_PQ_PATH, GOSEE_NPZ_PATH,
        CP_NPZ_MAIN, CP_NPZ_ALT, CP_LBL_PARQ, CP_LBL_JSONL,
        INC_PQ_PATH, INC_NPZ_PATH, INC_JSONL_PATH,
        # ... (WS/PREC se você já tinha)
    )
except Exception:
    SPH_PQ_PATH = SPH_NPZ_PATH = None
    GOSEE_PQ_PATH = GOSEE_NPZ_PATH = None
    CP_NPZ_MAIN = CP_NPZ_ALT = CP_LBL_PARQ = CP_LBL_JSONL = None
    INC_PQ_PATH = INC_NPZ_PATH = INC_JSONL_PATH = None

''' @st.cache_data(show_spinner=False)
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
        return None '''

@st.cache_data(show_spinner=False)
def _load_parquet(path: Path | None):
    if not path or not Path(path).exists(): return None
    try: return pd.read_parquet(path)
    except Exception: return None

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
    """Carrega Sphera + embeddings e devolve (df_sph, E_sph) alinhados por posição."""
    df = _load_parquet(SPH_PQ_PATH)
    E  = _load_npz_embeddings_any(SPH_NPZ_PATH)
    if df is None or E is None:
        return df, E
    # reset para garantir que o .iloc (posicional) case com E
    df = df.reset_index(drop=True)
    # se tamanhos divergirem, recorta para o mínimo (sem reordenar)
    n = min(len(df), E.shape[0])
    if len(df) != n:
        df = df.iloc[:n].reset_index(drop=True)
    if E.shape[0] != n:
        E = E[:n, :]
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
f _load_npz_embeddings_any(path: Path | None):
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
def load_dicts():
    # WS/PREC (se já possui)…
    E_cp   = _load_npz_embeddings_any(CP_NPZ_MAIN) or _load_npz_embeddings_any(CP_NPZ_ALT)
    L_cp   = _load_parquet(CP_LBL_PARQ) or _load_jsonl(CP_LBL_JSONL)
    return (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp)  # mantenha sua assinatura atual

@st.cache_data(show_spinner=False)
def _load_labels_any(parquet_path: Path, jsonl_path: Path):
    import json
    if parquet_path and parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if jsonl_path and jsonl_path.exists():
        try:
            rows = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            return pd.DataFrame(rows)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
def load_gosee():
    df = _load_parquet(GOSEE_PQ_PATH)
    E  = _load_npz_embeddings_any(GOSEE_NPZ_PATH)
    if df is None or E is None:
        return df, E
    df = df.reset_index(drop=True)
    n = min(len(df), E.shape[0])
    if len(df) != n:
        df = df.iloc[:n].reset_index(drop=True)
    if E.shape[0] != n:
        E = E[:n, :]
    return df, E

@st.cache_data(show_spinner=False)
def load_incidents():
    """Usa history_embeddings.npz + history_texts.jsonl (se houver)."""
    E = _load_npz_embeddings_any(INC_NPZ_PATH)
    df = _load_parquet(INC_PQ_PATH) or _load_jsonl(INC_JSONL_PATH)
    if df is None or E is None:
        return df, E
    df = df.reset_index(drop=True)
    n = min(len(df), E.shape[0])
    if len(df) != n: df = df.iloc[:n].reset_index(drop=True)
    if E.shape[0] != n: E = E[:n, :]
    return df, E

@st.cache_data(show_spinner=False)
def _load_jsonl(path: Path | None, text_key: str | None = None):
    if not path or not Path(path).exists(): return None
    try:
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        df = pd.DataFrame(rows)
        return df
    except Exception:
        return None
