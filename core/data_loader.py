
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
        WS_NPZ, WS_LBL_PARQ, WS_LBL_JSONL,
        PREC_NPZ, PREC_LBL_PARQ, PREC_LBL_JSONL,
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
def _load_npz_embeddings_any(npz_path: Path | str):
    """Carrega embeddings de um .npz aceitando chaves comuns ('embeddings', 'E', 'arr_0').
    Retorna np.ndarray ou None. NÃO usa `if E:` para evitar ambiguidade NumPy.
    """
    try:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            st.error(f"[RAG] NPZ não encontrado: {npz_path}")
            return None

        data = np.load(npz_path)
        for key in ("embeddings", "E", "arr_0"):
            if key in data.files:
                E = data[key]
                # normaliza dtype; não normaliza vetores aqui (fica para a busca)
                if E.dtype not in (np.float32, np.float64):
                    E = E.astype(np.float32)
                # sanity básica de 2D
                if E.ndim != 2:
                    st.error(f"[RAG] Embeddings com ndim={E.ndim}, esperado 2 em {npz_path}")
                    return None
                return E

        st.error(f"[RAG] Chaves esperadas não encontradas em {npz_path}. Encontradas: {list(data.files)}")
        return None

    except Exception as e:
        st.error(f"[RAG] Falha ao ler NPZ {npz_path}: {e}")
        return None


# ---------- (1) Sphera ----------
'''@st.cache_data(show_spinner=False)
def load_sphera():
    """Carrega Sphera (parquet + embeddings .npz) e alinha os comprimentos."""
    df = _load_parquet(SPH_PQ_PATH)
    E  = _load_npz_embeddings_any(SPH_NPZ_PATH)
    if df is None or E is None:
        return df, E
    df = df.reset_index(drop=True)
    n = min(len(df), E.shape[0])
    if len(df) != n:
        df = df.iloc[:n].reset_index(drop=True)
    if E.shape[0] != n:
        E = E[:n, :]
    return df, E  '''


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
def _load_labels_any(parquet_path: Path | None, jsonl_path: Path | None):
    return _load_parquet(parquet_path) or _load_jsonl(jsonl_path)

@st.cache_data(show_spinner=False)
def load_dicts():
    """Retorna: (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp). Tolerante a ausências."""
    E_ws   = _load_npz_embeddings_any(WS_NPZ)
    L_ws   = _load_labels_any(WS_LBL_PARQ, WS_LBL_JSONL)
    E_prec = _load_npz_embeddings_any(PREC_NPZ)
    L_prec = _load_labels_any(PREC_LBL_PARQ, PREC_LBL_JSONL)
    E_cp   = _load_npz_embeddings_any(CP_NPZ_MAIN) or _load_npz_embeddings_any(CP_NPZ_ALT)
    L_cp   = _load_labels_any(CP_LBL_PARQ, CP_LBL_JSONL)
    return E_ws, L_ws, E_prec, L_prec, E_cp, L_cp

try:
    from config import (
        GOSEE_PQ_PATH, GOSEE_NPZ_PATH,      # GoSee
        INC_PQ_PATH, INC_NPZ_PATH, INC_JSONL_PATH,  # Investigations/History
    )
except Exception:
    GOSEE_PQ_PATH = GOSEE_NPZ_PATH = None
    INC_PQ_PATH = INC_NPZ_PATH = INC_JSONL_PATH = None


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
    """Carrega Sphera (parquet + npz). Sempre retorna (DataFrame, np.ndarray|None).
    Faz checagens de alinhamento e evita NameError em caminhos de exceção.
    """
    df = None
    E = None
    try:
        # Parquet
        if Path(SPH_PQ_PATH).exists():
            df = pd.read_parquet(SPH_PQ_PATH)
        else:
            st.error(f"[RAG] Parquet do Sphera não encontrado: {SPH_PQ_PATH}")

        # NPZ
        E = _load_npz_embeddings_any(SPH_NPZ_PATH)  # já lida com erros/None

        # Sanidade de alinhamento
        if df is not None and E is not None:
            n_df = len(df)
            n_E  = E.shape[0]
            if n_df != n_E:
                st.warning(f"[RAG] Desalinhamento Sphera: df={n_df} linhas, E={n_E} vetores. "
                           f"Usando o mínimo para evitar estouro.")
                n = min(n_df, n_E)
                df = df.iloc[:n].reset_index(drop=True)
                E  = E[:n, :]

        # Retorno seguro (nunca levanta NameError)
        return (df if df is not None else pd.DataFrame(), E)

    except Exception as e:
        st.error(f"[RAG] Falha no load_sphera(): {e}")
        # Garante retorno consistente mesmo em erro
        return (df if df is not None else pd.DataFrame(), E)
