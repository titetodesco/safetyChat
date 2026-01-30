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

def _to_path(p):
    """Aceita Path | str | None e retorna Path ou None, sem explodir."""
    if isinstance(p, Path):
        return p
    if isinstance(p, str) and p.strip():
        return Path(p)
    return None

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
    """Carrega um .npz de embeddings. Retorna ndarray ou None.
    Aceita formatos com chave 'arr_0' (np.savez) ou nomeadas."""
    if not path or not isinstance(path, Path):
        st.warning("[RAG] Caminho NPZ inválido (None ou não-Path).")
        return None
    if not path.exists():
        st.warning(f"[RAG] NPZ não encontrado: {path}")
        return None

    import numpy as np
    try:
        data = np.load(path, allow_pickle=False)
        # tenta algumas chaves comuns
        for key in ("arr_0", "embeddings", "E", "data"):
            if key in data:
                arr = data[key]
                try:
                    return np.array(arr, dtype=float)
                except Exception:
                    return np.array(arr)
        # se vier múltiplas chaves, tenta a primeira que seja 2D
        for key in data.files:
            arr = data[key]
            try:
                arr_np = np.array(arr, dtype=float)
            except Exception:
                arr_np = np.array(arr)
            if arr_np.ndim >= 2:
                return arr_np
        # último recurso: retorna o primeiro array que encontrar
        first_key = data.files[0] if data.files else None
        return np.array(data[first_key]) if first_key else None
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
def _load_labels_any(parquet_path: Path | None, jsonl_path: Path | None):
    """Carrega rótulos (labels) de um parquet OU de um jsonl, de forma explícita.
    Nunca use 'df1 or df2' com DataFrame (ambíguo em pandas)."""
    # tenta parquet
    if parquet_path and parquet_path.exists():
        try:
            import pandas as pd
            return pd.read_parquet(parquet_path)
        except Exception as e:
            st.warning(f"[RAG] Falha ao ler PARQUET {parquet_path}: {e}")
    # fallback para jsonl
    if jsonl_path and jsonl_path.exists():
        try:
            import pandas as pd, json
            rows = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return pd.DataFrame(rows)
        except Exception as e:
            st.warning(f"[RAG] Falha ao ler JSONL {jsonl_path}: {e}")
    return None


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
    """Retorna (df_sphera, E_sphera). Nunca dá NameError, sempre retorna tupla."""
    from config import SPH_PQ_PATH, SPH_NPZ_PATH
    import pandas as pd

    df = pd.DataFrame()
    E  = None

    # Carrega parquet (se existir)
    if isinstance(SPH_PQ_PATH, Path) and SPH_PQ_PATH.exists():
        try:
            df = pd.read_parquet(SPH_PQ_PATH)
        except Exception as e:
            st.error(f"[RAG] Falha ao ler {SPH_PQ_PATH}: {e}")
    else:
        st.warning(f"[RAG] Parquet Sphera inexistente: {SPH_PQ_PATH}")

    # Carrega embeddings (se existir)
    if isinstance(SPH_NPZ_PATH, Path) and SPH_NPZ_PATH.exists():
        E = _load_npz_embeddings_any(SPH_NPZ_PATH)
    else:
        st.warning(f"[RAG] Embeddings Sphera inexistente: {SPH_NPZ_PATH}")

    # Diagnóstico enxuto
    st.write({
        "RAG_CHECK": {
            "sphera_parquet": str(SPH_PQ_PATH),
            "sphera_npz": str(SPH_NPZ_PATH),
            "len_df_sph": len(df) if not df.empty else 0,
            "E_is_array": (E is not None),
        }
    })

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
