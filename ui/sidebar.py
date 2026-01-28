
from __future__ import annotations
import streamlit as st
from typing import List, Tuple
import pandas as pd
from core.sphera import get_sphera_location_col, location_options

def render_prompts_selector(prompts_bank: str):
    st.sidebar.subheader("Assistente de Prompts")
    col1, col2 = st.sidebar.columns(2)
    txt_opts = ["(1) Síntese focada em lições", "(2) Recomendações", "(3) Padrões recorrentes", "(4) Análise causal", "(5) Resumo executivo"]
    upl_opts = ["(1) Destacar achados do upload", "(2) Comparar com Sphera", "(3) Lacunas e riscos", "(4) Checklists", "(5) Perguntas ao time"]

    sel_text = col1.selectbox("Texto", txt_opts, index=0)
    sel_upl  = col2.selectbox("Upload", upl_opts, index=0)
    load_btn = st.sidebar.button("Carregar no rascunho", use_container_width=True)
    return (sel_text, sel_upl, load_btn)

def render_retrieval_controls():
    st.sidebar.subheader("Recuperação – Sphera")
    k_sph = st.sidebar.slider("Top-K Sphera", 5, 100, 20, step=5)
    thr_sph = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01)
    years = st.sidebar.slider("Últimos N anos", 0, 10, 3, 1)
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
    agg_mode = st.sidebar.selectbox("Agregação", ["max","mean"], index=0)
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
    clear_upl = st.sidebar.button("Limpar uploads", use_container_width=True)
    clear_chat = st.sidebar.button("Limpar chat", use_container_width=True)
    return clear_upl, clear_chat
