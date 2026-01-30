# core/sphera.py
from __future__ import annotations
import os
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
    # Normaliza linhas de M (n_samples x dim)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / norms

def _cosine_query_vs_matrix(q: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Retorna array 1D com as similaridades cos entre q e cada linha de E.
    Assume E com shape (n, d). Normaliza q e E para coslínea.
    """
    if q.ndim != 1:
        q = q.reshape(-1)
    qn = _l2_normalize_vec(q)
    En = _l2_normalize_mat(E)
    # cos = En @ qn  (pois linhas de En são vetores L2-normalizados)
    return En @ qn


# --------------------------- API pública (usada pelo app) --------------------

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detecta a coluna 'LOCATION' (ou variações) que existe no DF.
    Retorna o nome da coluna ou None se não houver.
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
    e retorna o DF filtrado, preservando o alinhamento com embeddings via _rowid.
    """
    if df is None or df.empty:
        return df

    out = df

    # Filtro por locations (se existir coluna de local detectável)
    loc_col = get_sphera_location_col(out)
    if locations and loc_col and loc_col in out.columns:
        out = out[out[loc_col].astype(str).isin(locations)]

    # Filtro por substring em Description
    if substr:
        desc_col = "Description" if "Description" in out.columns else None
        if desc_col:
            mask = out[desc_col].astype(str).str.contains(substr, case=False, na=False)
            out = out[mask]

    # Filtro por últimos N anos, se houver uma coluna de data/ano
    # (ajuste este bloco ao seu esquema real; aqui tentamos heurística)
    if years and years > 0:
        year_col = None
        for c in ["year", "Year", "Ano", "ano", "EventYear"]:
            if c in out.columns:
                year_col = c
                break
        if year_col:
            # Mantém apenas registros dos últimos `years` anos
            try:
                max_year = pd.to_numeric(out[year_col], errors="coerce").dropna().astype(int).max()
                min_year = max_year - years + 1
                out = out[pd.to_numeric(out[year_col], errors="coerce").fillna(-1).astype(int) >= min_year]
            except Exception:
                pass

    return out


def topk_similar(
    query_text: str,
    df_base: pd.DataFrame,
    E_all: Optional[np.ndarray],
    topk: int = 20,
    min_cos: float = 0.30,
    text_col: str = "Description",
) -> List[dict]:
    """
    Calcula top-k mais similares ao `query_text` na base `df_base`,
    usando a matriz de embeddings `E_all` (alinhada ao DF original via `_rowid`).

    Retorno: lista de dicts com:
      - "idx": índice (inteiro) relativo ao df_base (ou _rowid do DF original, se disponível)
      - "cos": similaridade cosseno
      - "row": a linha do df_base (pd.Series) para consumo em tabelas/MD
    """
    # Sanidade
    if not query_text or not query_text.strip():
        return []
    if df_base is None or df_base.empty:
        return []
    if E_all is None or not isinstance(E_all, np.ndarray) or E_all.size == 0:
        return []

    # Precisamos alinhar DF filtrado às linhas correspondentes de E_all.
    # Convencionalmente, seu pipeline já mantém uma coluna `_rowid`
    # que indexa a linha original que gerou cada embedding.
    if "_rowid" not in df_base.columns:
        # Sem _rowid, não conseguimos mapear para E_all corretamente.
        # Fallback: assume que df_base está no mesmo índice que o DF original de E_all,
        # o que normalmente não é seguro após filtros. Preferimos falhar suavemente:
        st.warning("Base filtrada não possui coluna `_rowid`. Não é possível alinhar aos embeddings com segurança.")
        return []

    # Seleciona embeddings das linhas filtradas
    rowids = df_base["_rowid"].astype(int).to_numpy()
    max_id = int(rowids.max()) if len(rowids) else -1
    if max_id >= E_all.shape[0]:
        st.error("Há _rowid fora do intervalo da matriz de embeddings. Verifique a criação do _rowid e o alinhamento.")
        return []

    E_subset = E_all[rowids, :]

    # Embedding da consulta (usando o seu encoder padrão)
    try:
        encoder = ensure_st_encoder()  # mantém encoder no cache da sessão
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
    order = np.argsort(-sims)  # maiores primeiro
    hits = []
    for pos in order:
        cos = float(sims[pos])
        if cos < min_cos:
            continue
        row = df_base.iloc[int(pos)]
        hits.append({
            "idx": int(row.get("_rowid", pos)),
            "cos": cos,
            "row": row,  # mantém a linha para tabelas/contexto
        })
        if len(hits) >= topk:
            break

    return hits
