
from __future__ import annotations
import streamlit as st
import pandas as pd

from .ui.sidebar import (
    render_prompts_selector, render_retrieval_controls, render_advanced_filters,
    render_aggregation_controls, render_util_buttons
)
from .ui.main import render_main
from .core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from .core.dictionaries import aggregate_dict_matches_over_hits
from .core.context_builder import hits_dataframe, build_dic_matches_md, build_sphera_context_md
from .core.data_loader import load_dicts
from .services.llm_client import chat

st.set_page_config(page_title="SAFETY • CHAT — ESO", layout="wide")

go_btn, user_text, df_sph, E_sph, datasets_ctx, prompts_md, upl_texts = render_main()

# Sidebar (after main returns components)
sel_text, sel_upl, load_to_draft = render_prompts_selector(prompts_bank=prompts_md)
if load_to_draft:
    st.session_state.draft_prompt = (st.session_state.draft_prompt + "\n" + sel_text + "\n" + sel_upl).strip()

k_sph, thr_sph, years = render_retrieval_controls()
locations, substr, loc_col, loc_opts = render_advanced_filters(df_sph)
(agg_mode, per_event_thr, support_min, thr_ws, thr_prec, thr_cp, top_ws, top_prec, top_cp) = render_aggregation_controls()
clear_upl, clear_chat_btn = render_util_buttons()
if clear_upl:
    st.session_state.upld_texts = []
if clear_chat_btn:
    st.session_state.chat = []

# Only run when user clicks
if go_btn:
    # 1) Retrieve Sphera hits
    df_base = filter_sphera(df_sph, locations, substr, years)
    hits = topk_similar(user_text or st.session_state.draft_prompt, df_base, E_sph, k_sph, thr_sph)
    loc_col = get_sphera_location_col(df_sph)

    # 2) Aggregate dictionaries over hits
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
    dic_res, debug_raw = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        per_event_thr=per_event_thr, support_min=support_min, agg_mode=agg_mode,
        thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
        top_ws=top_ws, top_prec=top_prec, top_cp=top_cp
    )

    # 3) Build UI tables
    st.subheader(f"Eventos do Sphera (Top-{min(k_sph, len(hits))})")
    df_hits = hits_dataframe(hits, loc_col)
    st.dataframe(df_hits, use_container_width=True, hide_index=True)

    # 4) Debug expander
    from .ui.tables import show_debug_raw
    show_debug_raw(debug_raw)

    # 5) Compose context for LLM
    ctx_lines = [
        datasets_ctx,
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])

    messages = [
        {"role":"system", "content":"Você é o SAFETY • CHAT. Baseie-se no contexto fornecido e nas regras da organização para ESO."},
        {"role":"user", "content": (st.session_state.draft_prompt or user_text or "").strip()},
        {"role":"user", "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
    ]

    try:
        res = chat(messages, stream=False)
        content = res.get("message",{}).get("content","(sem conteúdo)")
    except Exception as e:
        content = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(content)

    st.session_state.chat.append({"role":"assistant","content":content})

# History (last 10)
if st.session_state.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in st.session_state.chat[-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content",""))
