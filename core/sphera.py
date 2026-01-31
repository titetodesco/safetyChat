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

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    """Descobre a melhor coluna de localização (se existir)."""
    if df is None or df.empty:
        return None
    for c in ["LOCATION", "Location", "local", "Local", "Area", "Instalação"]:
        if c in df.columns:
            return c
    return None


def filter_sphera(
    df: pd.DataFrame,
    locations: List[str] | None,
    substr: str | None,
    years: int | None
) -> pd.DataFrame:
    """Aplica filtros por local, substring em Description e janela de anos."""
    if df is None or df.empty:
        return df

    out = df

    # filtro por local
    loc_col = get_sphera_location_col(out)
    if locations and loc_col and loc_col in out.columns:
        out = out[out[loc_col].astype(str).isin(locations)]

    # filtro por substring em Description
    if substr and "Description" in out.columns:
        mask = out["Description"].astype(str).str.contains(substr, case=False, na=False)
        out = out[mask]

    # filtro por últimos N anos (heurística)
    if years and years > 0:
        year_col = None
        for c in ["year", "Year", "Ano", "ano", "EventYear"]:
            if c in out.columns:
                year_col = c
                break
        if year_col:
            try:
                yr = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
                max_year = int(yr.dropna().max())
                min_year = max_year - years + 1
                out = out[yr >= min_year]
            except Exception:
                pass

    return out


def topk_similar(
    query_text: str,
    df_base: pd.DataFrame,
    E_all: np.ndarray | None,
    *,
    topk: int = 20,
    min_sim: float = 0.30,
    text_col: str = "Description",
):
    """
    Retorna lista de dicts com:
      - idx: _rowid original (int)
      - cos: similaridade (float)
      - row: linha (pd.Series)

    Observações:
    - Requer que df_base tenha a coluna '_rowid' (alinhada com E_all).
    - E_all é a matriz completa de embeddings alinhada ao DF original.
    - Este método cria um subset E_subset usando os _rowid presentes em df_base.
    """
    # sanity checks
    if not query_text or not query_text.strip():
        return []
    if df_base is None or df_base.empty:
        return []
    if E_all is None or not isinstance(E_all, np.ndarray) or E_all.size == 0:
        return []
    if "_rowid" not in df_base.columns:
        st.warning("Base filtrada não possui coluna `_rowid` para alinhar com embeddings.")
        return []

    rowids = df_base["_rowid"].astype(int).to_numpy()
    if rowids.size == 0:
        return []
    if int(rowids.max()) >= E_all.shape[0]:
        st.error("Há _rowid fora do intervalo da matriz de embeddings. Verifique o alinhamento DF <-> embeddings.")
        return []

    # subset de embeddings pela indexação original
    E_subset = E_all[rowids, :]

    # cria/recupera encoder com o nome do modelo definido no config
    try:
        encoder = ensure_st_encoder(model_name=EMBED_MODEL_NAME)
    except TypeError:
        # compat com versões antigas que recebem só posicional
        encoder = ensure_st_encoder(EMBED_MODEL_NAME)
    except Exception as ex:
        st.error(f"Falha ao inicializar o encoder '{EMBED_MODEL_NAME}': {ex}")
        return []

    # embedding da consulta
    try:
        # IMPORTANTE: encode_query NÃO deve passar show_progress_bar ou kwargs não suportados
        q = encode_query(query_text, encoder)
    except Exception as ex:
        st.error(f"Falha ao gerar embedding da consulta: {ex}")
        return []

    # similaridades
    try:
        sims = _cosine_query_vs_matrix(q, E_subset)
    except Exception as ex:
        st.error(f"Falha ao calcular similaridades: {ex}")
        return []

    order = np.argsort(-sims)
    hits = []
    for pos in order:
        cos = float(sims[pos])
        if cos < min_sim:
            continue
        # cuidado: pos aqui é índice relativo ao df_base filtrado
        row = df_base.iloc[int(pos)]
        hits.append({
            "idx": int(row.get("_rowid", pos)),
            "cos": cos,
            "row": row,
        })
        if len(hits) >= topk:
            break

    return hits
