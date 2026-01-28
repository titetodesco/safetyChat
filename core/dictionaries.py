
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

def _labels_col(df: pd.DataFrame) -> str:
    for c in ["label","LABEL","term","Term","descricao","description","DESC"]:
        if c in df.columns:
            return c
    return df.columns[0]

def _family_name(name: str) -> str:
    return {"ws":"WS","prec":"Precursor","cp":"CP"}.get(name, name)

def _aggregate(scores: np.ndarray, mode: str) -> float:
    if scores.size == 0:
        return 0.0
    if mode == "mean":
        return float(scores.mean())
    return float(scores.max())  # default max

def aggregate_dict_matches_over_hits(
    hits: List[Tuple[str,float,pd.Series]],
    E_ws: Optional[np.ndarray], L_ws: Optional[pd.DataFrame],
    E_prec: Optional[np.ndarray], L_prec: Optional[pd.DataFrame],
    E_cp: Optional[np.ndarray], L_cp: Optional[pd.DataFrame],
    per_event_thr: float = 0.15,
    support_min: int = 1,
    agg_mode: str = "max",
    thr_ws: float = 0.30, thr_prec: float = 0.30, thr_cp: float = 0.30,
    top_ws: int = 10, top_prec: int = 10, top_cp: int = 10
):
    # Build a matrix V_desc from hit descriptions using SBERT (for dictionary match)
    if not hits:
        return {}, {}
    descs = [str(h[2].get("Description","")) for h in hits]
    from .encoding import ensure_st_encoder, encode_texts
    import os
    enc = ensure_st_encoder(os.getenv("ST_MODEL_NAME","sentence-transformers/all-MiniLM-L6-v2"))
    V_desc = encode_texts(enc, descs)

    results = {}
    debug = {}

    def compute_family(E_bank, L_bank, fam_key: str, thr_global: float, topN: int):
        if E_bank is None or L_bank is None or len(L_bank)==0:
            return []
        # cosine scores term vs each description
        S = (E_bank @ V_desc.T)  # [terms x events]
        valid = []
        for i in range(S.shape[0]):
            ev_idx = np.where(S[i, :] >= per_event_thr)[0]
            if ev_idx.size < support_min:
                continue
            score = _aggregate(S[i, ev_idx], agg_mode)
            if score >= thr_global:
                lab = str(L_bank.iloc[i].get(_labels_col(L_bank), f"TERM_{i}"))
                valid.append((lab, float(score)))
        valid.sort(key=lambda t: t[1], reverse=True)
        return valid[:topN]

    res_ws   = compute_family(E_ws, L_ws,   "ws",   thr_ws,   top_ws)
    res_prec = compute_family(E_prec, L_prec,"prec",thr_prec, top_prec)
    res_cp   = compute_family(E_cp, L_cp,   "cp",   thr_cp,   top_cp)

    results["WS"] = res_ws
    results["Precursores"] = res_prec
    results["CP"] = res_cp

    # debug lists without global thresholds (top raw)
    def debug_family(E_bank, L_bank, fam_key: str, topN: int = 20):
        out = []
        if E_bank is None or L_bank is None or len(L_bank)==0:
            return out
        S = (E_bank @ V_desc.T)
        for i in range(S.shape[0]):
            lab = str(L_bank.iloc[i].get(_labels_col(L_bank), f"TERM_{i}"))
            score = float(S[i, :].max())
            out.append((lab, score))
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:topN]
    debug["RAW_WS"]   = debug_family(E_ws, L_ws, "ws")
    debug["RAW_PREC"] = debug_family(E_prec, L_prec, "prec")
    debug["RAW_CP"]   = debug_family(E_cp, L_cp, "cp")

    return results, debug
