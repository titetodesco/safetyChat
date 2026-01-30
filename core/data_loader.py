from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import json

import urllib.request, os
#from config import SPH_NPZ_PATH, SPH_NPZ_URL
from config import SPH_PQ_PATH, SPH_NPZ_PATH
# import tolerante das constantes do config
try:
    from config import (
        SPH_PQ_PATH, SPH_NPZ_PATH,
        PROMPTS_MD_PATH, DATASETS_CONTEXT_PATH,
        # se você tiver os demais, mantêm aqui...
        WS_NPZ, WS_PARQ, WS_LBL_PARQ, WS_LBL_JSONL,
        PREC_NPZ, PREC_PARQ, PREC_LBL_PARQ, PREC_LBL_JSONL,
        CP_NPZ_MAIN, CP_NPZ_ALT, CP_LBL_PARQ, CP_LBL_JSONL,
        GOSEE_PQ_PATH, GOSEE_NPZ_PATH,
        INC_PQ_PATH, INC_NPZ_PATH, INC_JSONL_PATH,
    )
except Exception:
    SPH_PQ_PATH = SPH_NPZ_PATH = None
    PROMPTS_MD_PATH = DATASETS_CONTEXT_PATH = None
    WS_NPZ = WS_LBL_PARQ = WS_LBL_JSONL = None
    PREC_NPZ = PREC_LBL_PARQ = PREC_LBL_JSONL = None
    CP_NPZ_MAIN = CP_NPZ_ALT = CP_LBL_PARQ = CP_LBL_JSONL = None
    GOSEE_PQ_PATH = GOSEE_NPZ_PATH = None
    INC_PQ_PATH = INC_NPZ_PATH = INC_JSONL_PATH = None

@st.cache_data(show_spinner=False)
def _load_parquet(path: Path | None):
    if not path or not Path(path).exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_jsonl(path: Path | None):
    if not path or not Path(path).exists():
        return None
    try:
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        return pd.DataFrame(rows)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_npz_embeddings_any(path: Path | None):
    """
    Carrega um .npz que contenha embeddings.
    Retorna np.ndarray ou None. Nunca usa checagem booleana sobre arrays.
    """
    if path is None or not isinstance(path, Path):
        return None
    try:
        if not path.exists():
            st.warning(f"[RAG] NPZ não encontrado: {path}")
            return None
        data = np.load(path, allow_pickle=True)
        # Tente chaves comuns em ordem
        for key in ("embeddings", "E", "vectors", "arr_0"):
            if key in data.files:
                arr = data[key]
                # Garante 2D
                if arr is not None and arr.ndim == 1:
                    arr = np.stack(arr)
                return arr.astype(np.float32, copy=False)
        st.error(f"[RAG] NPZ {path} não contém chave de embeddings conhecida.")
        return None
    except Exception as e:
        st.error(f"[RAG] Falha ao ler NPZ {path}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_prompts_md(path: str | Path | None = None) -> str:
    """
    Lê o prompts.md do caminho fornecido OU do PROMPTS_MD_PATH do config.
    Aceita None (usa config), str ou Path.
    """
    p = Path(path) if path else (Path(PROMPTS_MD_PATH) if PROMPTS_MD_PATH else None)
    if not p or not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def load_datasets_context(path: str | Path | None = None) -> str:
    """
    Lê o datasets_context.md do caminho fornecido OU do DATASETS_CONTEXT_PATH do config.
    Aceita None (usa config), str ou Path.
    """
    p = Path(path) if path else (Path(DATASETS_CONTEXT_PATH) if DATASETS_CONTEXT_PATH else None)
    if not p or not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

# ---------- (3) Dicionários (seu loader atual pode chamar isto) ----------
@st.cache_data(show_spinner=False)
def _load_labels_any(parquet_path: Path | None, jsonl_path: Path | None) -> pd.DataFrame:
    """
    Carrega labels primeiro de PARQUET; se vazio/ausente, tenta JSONL.
    Retorna sempre um DataFrame (vazio se não achar).
    """
    df = None
    if parquet_path is not None and parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            st.error(f"[RAG] Falha ao ler labels PARQUET {parquet_path}: {e}")
            df = None

    if df is not None and not df.empty:
        return df

    if jsonl_path is not None and jsonl_path.exists():
        try:
            df = pd.read_json(jsonl_path, lines=True)
        except Exception as e:
            st.error(f"[RAG] Falha ao ler labels JSONL {jsonl_path}: {e}")
            df = None

    if df is None:
        df = pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def load_dicts():
    """
    Retorna (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp).

    Estratégia:
      - Embeddings: tenta NPZ; se None, tenta PARQUET (colunas embedding/vector/...).
      - Labels: PARQUET -> JSONL. Nunca usa 'or' entre DataFrames.
    """
    # WS
    E_ws = _load_npz_embeddings_any(WS_NPZ)
    if E_ws is None and WS_PARQ is not None:
        E_ws = _load_embeddings_from_parquet(WS_PARQ)
    L_ws = _load_labels_any(WS_LBL_PARQ, WS_LBL_JSONL)

    # PRECURSORES
    E_prec = _load_npz_embeddings_any(PREC_NPZ)
    if E_prec is None and PREC_PARQ is not None:
        E_prec = _load_embeddings_from_parquet(PREC_PARQ)
    L_prec = _load_labels_any(PREC_LBL_PARQ, PREC_LBL_JSONL)

    # CP
    E_cp = _load_npz_embeddings_any(CP_NPZ_MAIN)
    if E_cp is None:
        E_cp = _load_npz_embeddings_any(CP_NPZ_ALT)
    L_cp = _load_labels_any(CP_LBL_PARQ, CP_LBL_JSONL)

    return (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp)

@st.cache_data(show_spinner=False)
def load_gosee():
    """
    Carrega GoSee (parquet + embeddings .npz) e alinha comprimentos.
    Retorna: (df, E) ou (None, None) se ausente.
    """
    df = _load_parquet(GOSEE_PQ_PATH)
    E  = _load_npz_embeddings_any(GOSEE_NPZ_PATH)
    if df is None or E is None:
        return df, E
    df = df.reset_index(drop=True)
    n = min(len(df), getattr(E, "shape", (0, 0))[0])
    if len(df) != n:
        df = df.iloc[:n].reset_index(drop=True)
    if E.shape[0] != n:
        E = E[:n, :]
    return df, E

@st.cache_data(show_spinner=False)
def load_incidents():
    """
    Carrega Relatórios/Histórico: usa embeddings .npz e texto de parquet OU jsonl.
    Retorna: (df, E) ou (None, None) se ausente.
    """
    E = _load_npz_embeddings_any(INC_NPZ_PATH)
    # aceita parquet ou jsonl (preferência pelo parquet)
    df = _load_parquet(INC_PQ_PATH) or _load_jsonl(INC_JSONL_PATH)
    if df is None or E is None:
        return df, E
    df = df.reset_index(drop=True)
    n = min(len(df), getattr(E, "shape", (0, 0))[0])
    if len(df) != n:
        df = df.iloc[:n].reset_index(drop=True)
    if E.shape[0] != n:
        E = E[:n, :]
    return df, E

def _ensure_npz_local(npz_path: Path, url: str) -> bool:
    try:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz_path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, npz_path)
        return True
    except Exception as e:
        st.error(f"[RAG] Falha ao baixar NPZ de {url}: {e}")
        return False

@st.cache_data(show_spinner=False)

@st.cache_data(show_spinner=False)
def load_sphera():
    """
    Carrega a base Sphera e seus embeddings.
    - df: lido de SPH_PQ_PATH (parquet). Se ausente/erro -> DataFrame vazio.
    - E:  lido de SPH_NPZ_PATH (npz). Se ausente/erro -> None.
    Nunca usa checagem booleana sobre DataFrame/ndarray.
    """
    # -------- DF --------
    df = pd.DataFrame()
    try:
        if SPH_PQ_PATH is None or not isinstance(SPH_PQ_PATH, Path):
            st.error("[RAG] SPH_PQ_PATH é None ou não é Path. Confira config.py.")
        elif not SPH_PQ_PATH.exists():
            st.error(f"[RAG] Parquet inexistente: {SPH_PQ_PATH}")
        else:
            df = pd.read_parquet(SPH_PQ_PATH)
    except Exception as e:
        st.error(f"[RAG] Falha ao ler parquet {SPH_PQ_PATH}: {e}")
        df = pd.DataFrame()

    # -------- Embeddings --------
    E = None
    try:
        if SPH_NPZ_PATH is None or not isinstance(SPH_NPZ_PATH, Path):
            st.error("[RAG] SPH_NPZ_PATH é None ou não é Path. Confira config.py.")
        elif not SPH_NPZ_PATH.exists():
            st.error(f"[RAG] NPZ inexistente: {SPH_NPZ_PATH}")
        else:
            npz = np.load(SPH_NPZ_PATH, allow_pickle=True)
            for key in ("embeddings", "E", "vectors", "arr_0"):
                if key in npz.files:
                    arr = npz[key]
                    if arr is not None and arr.ndim == 1:
                        arr = np.stack(arr)
                    E = arr.astype(np.float32, copy=False)
                    break
            if E is None:
                st.error(f"[RAG] {SPH_NPZ_PATH} não contém chave de embeddings conhecida.")
    except Exception as e:
        st.error(f"[RAG] Falha ao ler NPZ {SPH_NPZ_PATH}: {e}")
        E = None

    return df, E

def _load_embeddings_from_parquet(parquet_path: Path,
                                  col_candidates=("embedding", "embeddings", "vector", "vectors", "emb")):
    """
    Se não houver NPZ, tenta carregar embeddings de um .parquet.
    A coluna deve conter listas/arrays por linha. Retorna np.ndarray ou None.
    """
    try:
        if parquet_path is None or not parquet_path.exists():
            return None
        dfp = pd.read_parquet(parquet_path)
        for c in col_candidates:
            if c in dfp.columns:
                vals = dfp[c].to_list()
                arr = np.array(vals, dtype=object)
                # “explode” a lista de listas em 2D
                if arr.ndim == 1:
                    arr = np.stack(arr).astype(np.float32)
                return arr
        st.warning(f"[RAG] {parquet_path} não tem coluna de vetor entre {col_candidates}.")
        return None
    except Exception as e:
        st.error(f"[RAG] Falha ao ler embeddings do parquet {parquet_path}: {e}")
        return None
