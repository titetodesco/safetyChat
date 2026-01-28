
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import streamlit as st

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def location_options(df: pd.DataFrame) -> List[str]:
    col = get_sphera_location_col(df) if df is not None else None
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
    vals = sorted(set(vals))
    return vals

def filter_sphera(df: pd.DataFrame, locations: List[str], substr: str, years: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    # time window
    if years and "EVENT_DATE" in out.columns:
        try:
            out["EVENT_DATE"] = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            out = out[out["EVENT_DATE"] >= cutoff]
        except Exception:
            pass
    # location
    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
        out = out[out[loc_col].astype(str).isin(set(locations))]
    # substring
    desc_col = "Description" if "Description" in out.columns else None
    if substr and desc_col:
        pat = re.escape(substr)
        out = out[out[desc_col].astype(str).str.contains(pat, case=False, na=False, regex=True)]
    return out

def topk_similar(query_text: str, df: pd.DataFrame, E: np.ndarray, topk: int, min_sim: float) -> List[Tuple[str,float,pd.Series]]:
    if not query_text or df is None or df.empty or E is None or E.size == 0:
        return []
    # Encoder is not required; we use precomputed E (already normalized) and assume
    # you computed the query vector elsewhere and passed as qv; but here we'll
    # compute a quick SBERT vector via a small cache to remain self-contained.
    from .encoding import ensure_st_encoder, encode_query
    import os
    model = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    enc = ensure_st_encoder(model)
    qv = encode_query(enc, query_text)
    # align E view to df index if numeric
    try:
        idx = df.index.to_numpy()
        if np.issubdtype(idx.dtype, np.integer):
            E_view = E[idx, :]
        else:
            E_view = E
    except Exception:
        E_view = E
    sims = (E_view @ qv).astype(float)
    ord_idx = np.argsort(-sims)
    out = []
    id_col = None
    for cand in ["Event ID", "EVENT_ID", "EVENTID"]:
        if cand in df.columns:
            id_col = cand; break
    for i in ord_idx[: max(1, int(topk)) ]:
        s = float(sims[i])
        if s < float(min_sim):
            continue
        row = df.iloc[i]
        evid = str(row.get(id_col, str(row.name)))
        out.append((evid, s, row))
    return out
