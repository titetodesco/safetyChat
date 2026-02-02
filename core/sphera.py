# core/sphera.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import os

try:
    from config import OLLAMA_EMBEDDING_MODEL as EMBED_MODEL_NAME
except Exception:
    try:
        from config import EMBEDDING_MODEL as EMBED_MODEL_NAME
    except Exception:
        EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

from core.encoding import ensure_st_encoder, encode_query


def _is_missing(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except Exception:
        return x is None


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def _extract_event_id(row: pd.Series, fallback: int) -> str:
    """
    Extrai um EventID real de uma linha do Sphera.
    Evita retornar índice/rowid como "EventID" quando a coluna não existe ou está vazia.
    """
    candidates = [
        "EventID", "EVENTID", "EVENT_ID", "Event ID", "EVENT ID", "ID", "Id", "id",
        "EventId", "EVENTId", "eventid", "event_id",
    ]
    for k in candidates:
        if k in row.index:
            v = row.get(k)
            if not _is_missing(v):
                return str(v).strip()

    # fallback: não temos EventID — devolve algo claro, mas não engana como EventID
    return f"ROW_{fallback}"


def get_sphera_location_col(df: pd.DataFrame | None) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]:
        if c in df.columns:
            return c
    return None


def filter_sphera(
    df: pd.DataFrame | None,
    locations: List[str] | None,
    substr: str | None,
    years: int | None
) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    loc_col = get_sphera_location_col(out)
    if locations and loc_col:
        out = out[out[loc_col].astype(str).isin(set(locations))]

    if substr and "Description" in out.columns:
        out = out[out["Description"].astype(str).str.contains(substr, case=False, na=False)]

    # Obs: filtro por "years" depende de existir coluna de data. Mantive como no seu código (não aplicava).
    return out if not out.empty else df


def topk_similar(
    query_text: str,
    df: pd.DataFrame,
    E: np.ndarray,
    topk: int = 20,
    min_sim: float = 0.30,
) -> List[Tuple[str, float, pd.Series]]:
    """
    Retorna lista de (EventID_str, cos_sim, row_series).
    IMPORTANTE: E precisa estar alinhado com df (mesma ordem e mesmo número de linhas).
    """
    if E is None or getattr(E, "size", 0) == 0:
        return []
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    # ✅ proteção contra desalinhamento
    if E.shape[0] != len(df):
        raise ValueError(
            f"[Sphera] Embeddings desalinhados: E.shape[0]={E.shape[0]} vs len(df)={len(df)}. "
            f"Filtre E junto com df (ex.: usando _rowid)."
        )

    model_name = os.getenv(
        "ST_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    enc = ensure_st_encoder(model_name)
    qv = encode_query(query_text, enc).astype(np.float32)
    qv = _l2_normalize_vec(qv)

    sims = (E @ qv).reshape(-1)
    idx = np.argsort(-sims)[: int(topk)]

    out: List[Tuple[str, float, pd.Series]] = []
    for i in idx:
        s = float(sims[i])
        if s < float(min_sim):
            continue
        row = df.iloc[int(i)]
        evid = _extract_event_id(row, int(i))
        out.append((evid, s, row))
    return out
