from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    # Pastas & docs
    DATASETS_CONTEXT_PATH, PROMPTS_MD_PATH,

    # Sphera
    SPH_PQ_PATH, SPH_NPZ_PATH,

    # GoSee
    GOSEE_PQ_PATH, GOSEE_NPZ_PATH,

    # Incidents / History
    INC_JSONL_PATH, INC_NPZ_PATH, INC_PQ_PATH,

    # Dicionários
    WS_NPZ, WS_LBL_PARQ, WS_LBL_JSONL,
    PREC_NPZ, PREC_LBL_PARQ, PREC_LBL_JSONL,
    CP_NPZ_MAIN, CP_NPZ_ALT, CP_LBL_PARQ, CP_LBL_JSONL,
)

# ---------------- Utilitários de IO ----------------

@st.cache_data(show_spinner=False)
def _load_parquet(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not isinstance(path, Path):
        return None
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_jsonl(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not isinstance(path, Path):
        return None
    if not path.exists():
        return None
    try:
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_npz_embeddings_strict(path: Optional[Path]) -> np.ndarray:
    """
    Carrega embeddings NUMPY (strict): falha claramente se ausente/invalid.
    Espera-se chave 'embeddings' OU array raiz em np.load(...).
    """
    if not isinstance(path, Path):
        raise FileNotFoundError("Caminho NPZ inválido (None ou não-Path).")
    if not path.exists():
        raise FileNotFoundError(f"NPZ não encontrado: {path}")

    with np.load(path, allow_pickle=False) as npz:
        # Se há uma única chave 'arr_0', tratamos como array raiz.
        if "embeddings" in npz.files:
            arr = npz["embeddings"]
        elif "arr_0" in npz.files and len(npz.files) == 1:
            arr = npz["arr_0"]
        else:
            # tenta achar a primeira chave com shape 2D
            candidates = [k for k in npz.files if np.array(npz[k]).ndim == 2]
            if not candidates:
                raise ValueError(f"NPZ {path} sem matriz 2D de embeddings.")
            arr = np.array(npz[candidates[0]])

    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError(f"Embeddings inválidos em {path} (esperado 2D).")
    return arr

# ---------------- Carregadores de Docs ----------------

@st.cache_data(show_spinner=False)
def load_datasets_context(path: Optional[Path]) -> Optional[str]:
    if isinstance(path, Path) and path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_prompts_md(path: Optional[Path]) -> Optional[str]:
    if isinstance(path, Path) and path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

# ---------------- Carregadores de Bases com Embeddings ----------------

@st.cache_data(show_spinner=False)
def load_sphera() -> Tuple[pd.DataFrame, np.ndarray | None]:
    df = _load_parquet(SPH_PQ_PATH)
    if df is None or df.empty:
        df = pd.DataFrame()
        E = None
        return df, E

    try:
        E = _load_npz_embeddings_strict(SPH_NPZ_PATH)
    except Exception as e:
        st.error(f"[RAG] Falha ao ler NPZ {SPH_NPZ_PATH}: {e}")
        return df, None

    # Ajuste de comprimento se necessário (cautela: mantém alinhamento 1:1 por índice)
    if len(df) != E.shape[0]:
        m = min(len(df), E.shape[0])
        df = df.iloc[:m].reset_index(drop=True)
        E = E[:m, :]

    return df, E

@st.cache_data(show_spinner=False)
def load_gosee() -> Tuple[pd.DataFrame, np.ndarray | None]:
    df = _load_parquet(GOSEE_PQ_PATH)
    if df is None or df.empty:
        return pd.DataFrame(), None

    try:
        E = _load_npz_embeddings_strict(GOSEE_NPZ_PATH)
    except Exception:
        return df, None

    if len(df) != E.shape[0]:
        m = min(len(df), E.shape[0])
        df = df.iloc[:m].reset_index(drop=True)
        E = E[:m, :]
    return df, E

@st.cache_data(show_spinner=False)
def load_incidents() -> Tuple[pd.DataFrame, np.ndarray | None]:
    # History: fonte primária em JSONL; se houver Parquet, pode-se usar também.
    df = None
    if isinstance(INC_PQ_PATH, Path):
        df = _load_parquet(INC_PQ_PATH)
    if df is None or df.empty:
        df = _load_jsonl(INC_JSONL_PATH)
    if df is None:
        return pd.DataFrame(), None

    try:
        E = _load_npz_embeddings_strict(INC_NPZ_PATH)
    except Exception:
        return df, None

    if len(df) != E.shape[0]:
        m = min(len(df), E.shape[0])
        df = df.iloc[:m].reset_index(drop=True)
        E = E[:m, :]
    return df, E

# ---------------- Rótulos de dicionários + Embeddings obrigatórios ----------------

@st.cache_data(show_spinner=False)
def _load_labels_any(parquet_path: Optional[Path], jsonl_path: Optional[Path]) -> pd.DataFrame:
    df = _load_parquet(parquet_path)
    if df is not None:
        return df
    df = _load_jsonl(jsonl_path)
    if df is not None:
        return df
    raise FileNotFoundError(f"Rótulos não encontrados (Parquet nem JSONL): {parquet_path} / {jsonl_path}")

@st.cache_data(show_spinner=False)
def load_dicts():
    """
    Retorna: (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp)
    Nesta aplicação, os NPZ são **obrigatórios** para WS, Precursores e CP.
    """
    # WS
    E_ws = _load_npz_embeddings_strict(WS_NPZ)
    L_ws = _load_labels_any(WS_LBL_PARQ, WS_LBL_JSONL)

    # Precursores
    E_prec = _load_npz_embeddings_strict(PREC_NPZ)
    L_prec = _load_labels_any(PREC_LBL_PARQ, PREC_LBL_JSONL)

    # CP
    cp_npz_path = CP_NPZ_MAIN if isinstance(CP_NPZ_MAIN, Path) else None
    if cp_npz_path is None and isinstance(CP_NPZ_ALT, Path):
        cp_npz_path = CP_NPZ_ALT
    E_cp = _load_npz_embeddings_strict(cp_npz_path)
    L_cp = _load_labels_any(CP_LBL_PARQ, CP_LBL_JSONL)

    # Ajustes defensivos de tamanho (se necessário)
    def _align(E: np.ndarray, L: pd.DataFrame):
        m = min(E.shape[0], len(L))
        if E.shape[0] != m or len(L) != m:
            return E[:m, :], L.iloc[:m].reset_index(drop=True)
        return E, L

    E_ws, L_ws   = _align(E_ws, L_ws)
    E_prec, L_prec = _align(E_prec, L_prec)
    E_cp, L_cp   = _align(E_cp, L_cp)

    return E_ws, L_ws, E_prec, L_prec, E_cp, L_cp
