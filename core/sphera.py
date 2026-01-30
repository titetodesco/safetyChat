# core/sphera.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st

from core.encoding import ensure_st_encoder, encode_query


# --------------------------- Utilidades internas -----------------------------

def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n

def _l2_normalize_mat(M: np.ndarray) -> np.ndarray:
    # Normaliza linhas de M (n amostras x dim)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / norms

def _cosine_query_vs_matrix(q: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Retorna array 1D com as similaridades cos entre q e cada linha de E.
    """
    if q.ndim != 1:
        q = q.reshape(-1)
    qn = _l2_normalize_vec(q)
    En = _l2_normalize_mat(E)
    return En @ qn


# --------------------------- API pública (usada pelo app) --------------------

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detecta a coluna de local (LOCATION/Location/Local/Area/Instalação...).
    """
    if df is None or df.empty:
        return None
    candidates = ["LOCATION", "Location", "local", "Local", "Area", "Instalação"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def filter_sphera(
    df: pd.DataFrame,
    locations: List[str] | None,
    substr: str | None,
    years: int | None
) -> pd.DataFrame:
    """
    Aplica filtros (location, substring em Description, janela de anos)
    e retorna o DF filtrado, preservando o alinhamento via coluna `_rowid`.
    """
    if df is None or df.empty:
        return df

    out = df

    # Filtro por locations (se existir coluna de local)
    loc_col = get_sphera_location_col(out)
    if locations and loc_col and loc_col in out.columns:
        out = out[out[loc_col].astype(str).isin(locations)]

    # Filtro por substring em Description
    if substr:
        if "Description" in out.columns:
            mask = out["Description"].astype(str).str.contains(substr, case=False, na=False)
            out = out[mask]

    # Filtro por últimos N anos (heurística — ajuste para o seu esquema real)
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
    E_all: Optional[np.ndarray],
    *,
    topk: int = 20,
    min_sim: float = 0.30,
    text_col: str = "Description",
) -> List[dict]:
    """
    Calcula top-k mais similares ao `query_text` dentro de `df_base`,
    usando a matriz de embeddings `E_all` (alinhada ao DF original via `_rowid`).

    Retorna lista de dicts:
      - "idx": índice original (_rowid) da linha
      - "cos": similaridade cosseno
      - "row": a linha do df_base (pd.Series)
    """
    # Sanidade
    if not query_text or not query_text.strip():
        return []
    if df_base is None or df_base.empty:
        return []
    if E_all is None or not isinstance(E_all, np.ndarray) or E_all.size == 0:
        return []

    if "_rowid" not in df_base.columns:
        st.warning("Base filtrada não possui coluna `_rowid`. Não é possível alinhar com embeddings.")
        return []

    rowids = df_base["_rowid"].astype(int).to_numpy()
    if len(rowids) == 0:
        return []

    max_id = int(rowids.max())
    if max_id >= E_all.shape[0]:
        st.error("Há _rowid fora do intervalo da matriz de embeddings. Verifique o alinhamento DF <-> embeddings.")
        return []

    # Subconjunto de embeddings
    E_subset = E_all[rowids, :]

    # Embedding da consulta com o seu encoder padrão
    try:
        encoder = ensure_st_encoder()
        q = encode_query(query_text, encoder)  # 1D np.ndarray
    except Exception as ex:
        st.error(f"Falha ao gerar embedding da consulta: {ex}")
        return []

    # Similaridades (cos)
    try:
        sims = _cosine_query_vs_matrix(q, E_subset)  # shape (n,)
    except Exception as ex:
        st.error(f"Falha ao calcular similaridades: {ex}")
        return []

    # Ordena por similaridade desc e aplica limiar
    order = np.argsort(-sims)
    hits = []
    for pos in order:
        cos = float(sims[pos])
        if cos < min_sim:
            continue
        row = df_base.iloc[int(pos)]
        hits.append({
            "idx": int(row.get("_rowid", pos)),
            "cos": cos,
            "row": row,
        })
        if len(hits) >= topk:
            break

    return hits
