from __future__ import annotations
import streamlit as st
import pandas as pd

from ui.sidebar import (
    render_prompts_selector, render_retrieval_controls, render_advanced_filters,
    render_aggregation_controls, render_util_buttons,
)
from ui.main import render_main

from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.dictionaries import aggregate_dict_matches_over_hits
from core.context_builder import (
    hits_dataframe, build_dic_matches_md, build_sphera_context_md,
)
from core.data_loader import load_dicts
from services.llm_client import chat
from ui.tables import show_debug_raw  # import absoluto

# ------------------------------------------------------------------------------
# Configuração inicial
# ------------------------------------------------------------------------------
st.set_page_config(page_title="SAFETY  CHAT ", layout="wide")

# Render do conteúdo principal (área central) e carregamentos base
go_btn, user_text, df_sph, E_sph, datasets_ctx, prompts_md, upl_texts = render_main()

# Estado base SEMPRE antes de widgets da sidebar
st.session_state.setdefault("draft_prompt", "")
st.session_state.setdefault("chat", [])
st.session_state.setdefault("upld_texts", [])

# Detecta coluna de Location e deixa disponível para a sidebar
try:
    loc_col_detected = get_sphera_location_col(df_sph) if df_sph is not None else None
except Exception:
    loc_col_detected = None
st.session_state["sphera_loc_col"] = loc_col_detected

# Constrói as listas de prompts (Texto/Upload) a partir de prompts_md (seu parser original pode estar em render_main)
# Aqui só garantimos que existam listas no session_state para o Assistente de Prompts:
st.session_state.setdefault("prompt_text_opts", st.session_state.get("prompt_text_opts", []))
st.session_state.setdefault("prompt_upl_opts",  st.session_state.get("prompt_upl_opts",  []))

# ------------------------------------------------------------------------------
# Sidebar – cada bloco uma ÚNICA vez
# ------------------------------------------------------------------------------
# 1) Assistente de Prompts
sel_text, sel_upl, load_to_draft = render_prompts_selector(prompts_bank=prompts_md, key_prefix="sb_")
if load_to_draft:
    base = (st.session_state.get("draft_prompt") or "").strip()
    parts = [base]
    if sel_text:
        parts.append(str(sel_text).strip())
    if sel_upl:
        parts.append(str(sel_upl).strip())
    st.session_state["draft_prompt"] = "\n\n".join([p for p in parts if p]).strip()
    st.rerun()

# 2) Recuperação – Sphera
k_sph, thr_sph, years = render_retrieval_controls()

# 3) Filtros avançados – Sphera
locations, substr, loc_col, loc_opts = render_advanced_filters(df_sph)

# 4) Agregação sobre eventos recuperados (Sphera)
(
    agg_mode, per_event_thr, support_min,
    thr_ws, thr_prec, thr_cp,
    top_ws, top_prec, top_cp
) = render_aggregation_controls()

# 5) Utilitários
clear_upl, clear_chat_btn = render_util_buttons()
if clear_upl:
    st.session_state["upld_texts"] = []
if clear_chat_btn:
    st.session_state["chat"] = []

# ------------------------------------------------------------------------------
# Execução: SOMENTE ao clicar no botão principal
# ------------------------------------------------------------------------------
if go_btn:
    user_input = (user_text or st.session_state.get("draft_prompt") or "").strip()
    if not user_input:
        st.warning("Escreva algo no 'Conteúdo do prompt' ou em 'Texto de análise (para Sphera)' antes de enviar.")
        st.stop()

    # 1) Recuperação (Sphera)
    df_base = filter_sphera(df_sph, locations, substr, years)
    hits = topk_similar(user_input, df_base, E_sph, k_sph, thr_sph)
    loc_col_effective = get_sphera_location_col(df_base or df_sph)

    with st.expander("🔧 Diagnóstico RAG", expanded=False):
        st.write({
            "len(df_sph)": 0 if df_sph is None else len(df_sph),
            "len(df_base)": 0 if df_base is None else len(df_base),
            "E_sph.shape": None if E_sph is None else tuple(E_sph.shape),
            "hits": len(hits),
            "k_sph": k_sph,
            "thr_sph": float(thr_sph),
            "loc_col_effective": loc_col_effective,
        })

    # 2) Agregação dicionários (Sphera hits -> WS/Precursores/CP)
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
    dic_res, debug_raw = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        per_event_thr=per_event_thr, support_min=support_min, agg_mode=agg_mode,
        thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
        top_ws=top_ws, top_prec=top_prec, top_cp=top_cp
    )

    # 3) Tabela de hits
    st.subheader(f"Eventos do Sphera (Top-{min(k_sph, len(hits))})")
    df_hits = hits_dataframe(hits, loc_col_effective)
    st.dataframe(df_hits, use_container_width=True, hide_index=True)

    # 4) Depuração dos dicionários
    show_debug_raw(debug_raw)

    # 5) Contexto ao LLM
    ctx_lines = [
        datasets_ctx,                                  # datasets_context.md (sempre injetado)
        build_sphera_context_md(hits, loc_col_effective),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])

    messages = [
        {"role": "system", "content": "Você é o SAFETY • CHAT. Baseie-se no contexto fornecido e nas regras da organização para ESO."},
        {"role": "user",   "content": user_input},
        {"role": "user",   "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
    ]

    try:
        res = chat(messages, stream=False)
        content = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        content = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(content)

    st.session_state["chat"].append({"role": "assistant", "content": content})

# Histórico (últimos 10)
if st.session_state.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in st.session_state["chat"][-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
