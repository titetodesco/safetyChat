# core/sphera.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import streamlit as st
import os
from core.encoding import ensure_st_encoder, encode_query
from services.embedding_client import embed_text  # <-- troque pelo seu módulo real

def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def topk_similar(
    query_text: str,
    df_base: pd.DataFrame,
    E_all: np.ndarray,
    topk: int = 20,
    min_cos: float = 0.30
) -> List[Tuple[int, float]]:
    """
    Retorna lista de (rowid, score) já ordenada por score desc, filtrando por min_cos.
    rowid é df_base['_rowid'] (posicional no E_all da base original).
    """
    if not query_text or df_base.empty:
        return []

    # 1) Embedding da consulta (mesmo modelo dos NPZ!)
    v = embed_text(query_text)  # deve retornar shape (d,)
    v = _l2_normalize_vec(v.astype(np.float32))

    # 2) Subconjunto E pelo _rowid presente em df_base
    if "_rowid" not in df_base.columns:
        raise KeyError("df_base não possui coluna '_rowid'. Use load_sphera() que injeta isso.")
    rowids = df_base["_rowid"].to_numpy(dtype=np.int64)
    E_sub = E_all[rowids]  # shape (n_base, d) — já L2-normalizado no loader

    # 3) Cosine = dot(E_sub, v)
    scores = (E_sub @ v).astype(np.float32)  # shape (n_base,)

    # 4) Ordena e aplica limiar
    ord_idx = np.argsort(-scores)
    ord_idx = ord_idx[:min(topk, len(ord_idx))]
    hits = []
    for j in ord_idx:
        s = float(scores[j])
        if s >= min_cos:
            hits.append((int(rowids[j]), s))

    return hits


def get_sphera_location_col(df: pd.DataFrame | None) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def location_options(df: pd.DataFrame | None) -> List[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    col = get_sphera_location_col(df)
    if not col:
        return []
    vals = (
        df[col].astype(str).fillna("").str.strip().replace({"": None}).dropna().unique().tolist()
    )
    return sorted(set(vals))

def filter_sphera(df: pd.DataFrame | None, locations: List[str], substr: str, years: int) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()

    # janela temporal (se houver)
    if years and "EVENT_DATE" in out.columns:
        try:
            out["EVENT_DATE"] = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            out = out[out["EVENT_DATE"] >= cutoff]
        except Exception:
            pass

    # location (se houver)
    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
        out = out[out[loc_col].astype(str).isin(set(locations))]

    # substring em Description (case-insensitive)
    if substr and "Description" in out.columns:
        pat = re.escape(substr)
        out = out[out["Description"].astype(str).str.contains(pat, case=False, na=False, regex=True)]

    # se ficou vazio, devolve base original (para não “matar” o RAG sem aviso)
    return out if not out.empty else df
