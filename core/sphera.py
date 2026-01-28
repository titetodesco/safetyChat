# core/sphera.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import streamlit as st

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

    # janela temporal (se existir a coluna)
    if years and "EVENT_DATE" in out.columns:
        try:
            out["EVENT_DATE"] = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            out = out[out["EVENT_DATE"] >= cutoff]
        except Exception:
            pass

    # filtro de location
    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
        out = out[out[loc_col].astype(str).isin(set(locations))]

    # filtro substring (case-insensitive) em Description, se existir
    if substr and "Description" in out.columns:
        pat = re.escape(substr)
        out = out[out["Description"].astype(str).str.contains(pat, case=False, na=False, regex=True)]

    return out
