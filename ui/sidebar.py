# ui/sidebar.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import List, Tuple

from core.sphera import get_sphera_location_col, location_options

def render_prompts_selector(*, prompts_bank: str, key_prefix: str = "sb_") -> Tuple[str|None, str|None, bool]:
    """
    Retorna (sel_text, sel_upl, load_to_draft).
    Usa keys únicas com prefixo para evitar duplicação.
    """
    with st.sidebar.expander("Assistente de Prompts", expanded=False):
        # Parse simples: sua lógica já existente que gera txt_opts e upl_opts
        txt_opts = [f"Prompt {i}" for i in range(1, 6)]
        upl_opts = [f"Prompt {i}" for i in range(1, 5+1)]

        # importante: keys únicas
        sel_text = st.selectbox(
            "Texto",
            options=txt_opts,
            index=None, placeholder="Selecione um prompt (Texto)",
            key=f"{key_prefix}sel_text_prompt",
        )
        sel_upl = st.selectbox(
            "Upload",
            options=upl_opts,
            index=None, placeholder="Selecione um prompt (Upload)",
            key=f"{key_prefix}sel_upl_prompt",
        )

        load_to_draft = st.button("Carregar no rascunho", key=f"{key_prefix}load_to_draft", use_container_width=True)

    return sel_text, sel_upl, load_to_draft

def render_retrieval_controls():
    st.sidebar.subheader("Recuperação – Sphera")
    k_sph  = st.sidebar.slider("Top-K Sphera", 5, 100, 20, step=5)
    thr_sph = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01)
    years  = st.sidebar.slider("Últimos N anos", 0, 10, 3, 1)
    return k_sph, thr_sph, years

def render_advanced_filters(df_sph: pd.DataFrame):
    st.sidebar.subheader("Filtros avançados – Sphera")
    loc_col = get_sphera_location_col(df_sph) if df_sph is not None else None
    loc_opts = location_options(df_sph) if df_sph is not None else []
    locations = st.sidebar.multiselect("Location", loc_opts, default=[])
    substr = st.sidebar.text_input("Description contém (substring)", value="")
    return locations, substr, loc_col, loc_opts

def render_aggregation_controls():
    st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")
    agg_mode = st.sidebar.selectbox("Agregação", ["max", "mean"], index=0)
    per_event_thr = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.15, 0.01)
    support_min = st.sidebar.number_input("Suporte mínimo (nº de eventos)", 1, 100, 1, 1)
    thr_ws   = st.sidebar.slider("Limiar de similaridade WS", 0.0, 1.0, 0.30, 0.01)
    thr_prec = st.sidebar.slider("Limiar de similaridade Precursor", 0.0, 1.0, 0.30, 0.01)
    thr_cp   = st.sidebar.slider("Limiar de similaridade CP", 0.0, 1.0, 0.30, 0.01)
    top_ws   = st.sidebar.slider("Top-N WS", 1, 50, 10, 1)
    top_prec = st.sidebar.slider("Top-N Precursores", 1, 50, 10, 1)
    top_cp   = st.sidebar.slider("Top-N CP", 1, 50, 10, 1)
    return (agg_mode, per_event_thr, support_min, thr_ws, thr_prec, thr_cp, top_ws, top_prec, top_cp)

def render_util_buttons():
    st.sidebar.subheader("Utilitários")
    clear_upl  = st.sidebar.button("Limpar uploads", use_container_width=True)
    clear_chat = st.sidebar.button("Limpar chat", use_container_width=True)
    return clear_upl, clear_chat
