from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional

import streamlit as st
import pandas as pd

# ------------------ Assistente de Prompts ------------------
def render_prompts_selector(prompts_bank: Dict[str, List[str]] | None):
    """
    Espera um dict:
      {"texto": [opções...], "upload": [opções...]}
    Retorna: (sel_text, sel_upl, load_to_draft)
    """
    banco = prompts_bank or {}
    txt_opts = banco.get("texto") or []
    upl_opts = banco.get("upload") or []

    with st.sidebar.expander("Assistente de Prompts", expanded=False):
        # keys únicas para não colidir com outros lugares
        sel_text = st.selectbox(
            "Texto",
            options=txt_opts,
            index=None,
            placeholder="Selecione um prompt (Texto)",
            key="sb_sel_text_prompt",
        )
        # nesta versão não usamos o bloco de upload para evitar confusão
        sel_upl = None

        load_to_draft = st.button("Carregar no rascunho", key="sb_btn_load_prompt", use_container_width=True)

    return sel_text, sel_upl, load_to_draft

# ------------------ Recuperação – Sphera -------------------
def render_retrieval_controls() -> Tuple[int, float, int]:
    st.sidebar.subheader("Recuperação – Sphera")
    k_sph  = st.sidebar.slider("Top-K Sphera", 5, 100, 20, step=5, key="sb_topk_sph")
    thr_sph = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01, key="sb_thr_sph")
    years  = st.sidebar.slider("Últimos N anos", 0, 10, 3, 1, key="sb_years")
    return k_sph, thr_sph, years

# ------------------ Filtros avançados ----------------------
def render_advanced_filters(df_sph: Optional[pd.DataFrame]):
    st.sidebar.subheader("Filtros avançados – Sphera")

    substr = st.sidebar.text_input("Description contém (substring)", key="sb_desc_contains")

    loc_col = None
    loc_options = []
    locations = []

    if isinstance(df_sph, pd.DataFrame) and not df_sph.empty:
        # tentar inferir coluna de location
        for c in ["LOCATION", "Location", "FPSO", "FPSO/Unidade", "Unidade", "Local"]:
            if c in df_sph.columns:
                loc_col = c
                break
        if loc_col:
            loc_options = sorted([str(x) for x in df_sph[loc_col].dropna().unique().tolist()])
            locations = st.sidebar.multiselect(
                f"Location (coluna: {loc_col})",
                options=loc_options,
                default=[],
                key="sb_locations",
            )
        else:
            st.sidebar.caption("Coluna de Location não encontrada (filtro desabilitado).")
    else:
        st.sidebar.caption("Base Sphera indisponível (filtro desabilitado).")

    return locations, substr, loc_col, loc_options

# ------------------ Agregação sobre hits -------------------
def render_aggregation_controls():
    st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")

    agg_mode = st.sidebar.selectbox("Agregação", options=["max", "mean"], index=0, key="sb_agg_mode")
    per_event_thr = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01, key="sb_per_event_thr")
    support_min = st.sidebar.number_input("Suporte mínimo (nº eventos)", min_value=1, value=2, step=1, key="sb_support_min")

    st.sidebar.markdown("---")
    thr_ws   = st.sidebar.slider("Limiar global WS", 0.0, 1.0, 0.30, 0.01, key="sb_thr_ws")
    thr_prec = st.sidebar.slider("Limiar global Precursores", 0.0, 1.0, 0.30, 0.01, key="sb_thr_prec")
    thr_cp   = st.sidebar.slider("Limiar global CP", 0.0, 1.0, 0.30, 0.01, key="sb_thr_cp")

    top_ws   = st.sidebar.number_input("Top-N WS", min_value=1, value=10, step=1, key="sb_top_ws")
    top_prec = st.sidebar.number_input("Top-N Precursores", min_value=1, value=10, step=1, key="sb_top_prec")
    top_cp   = st.sidebar.number_input("Top-N CP", min_value=1, value=10, step=1, key="sb_top_cp")

    return (agg_mode, per_event_thr, support_min, thr_ws, thr_prec, thr_cp,
            top_ws, top_prec, top_cp)

# ------------------ Utilidades ------------------------------
def render_util_buttons():
    st.sidebar.subheader("Utilidades")
    clear_upl  = st.sidebar.button("Limpar uploads", key="sb_btn_clear_upl", use_container_width=True)
    clear_chat = st.sidebar.button("Limpar chat",   key="sb_btn_clear_chat", use_container_width=True)
    return clear_upl, clear_chat
