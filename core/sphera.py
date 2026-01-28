# core/sphera.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import streamlit as st

# --- adicione em core/sphera.py ---
from core.encoding import ensure_st_encoder, encode_query  # no topo do arquivo, junto aos imports

def topk_similar(query_text: str, df: pd.DataFrame | None, E: np.ndarray | None,
                 topk: int = 20, min_sim: float = 0.30):
    if not query_text or df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if E is None or getattr(E, "size", 0) == 0:
        return []

    # vetor de consulta
    import os
    model_name = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    enc = ensure_st_encoder(model_name)
    qv = encode_query(enc, query_text)  # normalizado

    # ALINHAMENTO POR POSIÇÃO
    n = len(df)
    if E.shape[0] < n:
        n = E.shape[0]
        df = df.iloc[:n].reset_index(drop=True)
    E_view = E[:n, :]  # primeira n linhas

    sims = (E_view @ qv).astype(float)
    order = np.argsort(-sims)

    # coluna de id
    id_col = None
    for cand in ("Event ID","EVENT_ID","EVENTID","id","ID"):
        if cand in df.columns:
            id_col = cand; break

    out = []
    k = max(1, int(topk))
    for i in order[:k]:
        s = float(sims[i])
        if s < float(min_sim):
            continue
        row = df.iloc[i]
        evid = str(row.get(id_col, str(i)))
        out.append((evid, s, row))
    return out


def get_sphera_location_col(df: pd.DataFrame | None) -> Optional[str]:
    # guarda contra None e contra objetos não-DataFrame
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
        df[col]
        .astype(str)
        .fillna("")
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(set(vals))

def filter_sphera(df: pd.DataFrame | None, locations: List[str], substr: str, years: int) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()

    # Janela temporal: só se a coluna existir de verdade
    if years and "EVENT_DATE" in out.columns:
        try:
            out["EVENT_DATE"] = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            out = out[out["EVENT_DATE"] >= cutoff]
        except Exception:
            pass

    # Location: só filtra se o usuário escolheu algo + coluna disponível
    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
        out = out[out[loc_col].astype(str).isin(set(locations))]

    # Substring: só se Description existir
    if substr and "Description" in out.columns:
        pat = re.escape(substr)
        out = out[out["Description"].astype(str).str.contains(pat, case=False, na=False, regex=True)]

    # Evitar retornar vazio sem o usuário perceber
    if out.empty:
        return df  # fallback: devolve base original para não zerar o RAG

    return out

