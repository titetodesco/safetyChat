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
    build_gosee_context_md, build_investigation_context_md,   # <- importe os novos
)
from core.data_loader import load_dicts
from services.llm_client import chat
from ui.tables import show_debug_raw  # <- import absoluto (evita erro de import)
from core.data_loader import load_gosee, load_incidents
from core.sphera import topk_similar  # já existente

# Carrega as outras bases
df_gosee, E_gosee = load_gosee()
df_inc, E_inc = load_incidents()
if df_inc is not None and E_inc is not None:
    hits_inc = topk_similar(user_input, df_inc, E_inc, topk=k_sph, min_sim=thr_sph)
    ctx_lines.append(build_investigation_context_md(hits_inc))


st.set_page_config(page_title="SAFETY  CHAT ", layout="wide")

# Recupera similares (sem filtros avançados, a menos que você deseje algum específico)
hits_gosee = topk_similar(user_input, df_gosee, E_gosee, topk=k_sph, min_sim=thr_sph) if E_gosee is not None else []
hits_inc   = topk_similar(user_input, df_inc,   E_inc,   topk=k_sph, min_sim=thr_sph) if E_inc   is not None else []

go_btn, user_text, df_sph, E_sph, datasets_ctx, prompts_md, upl_texts = render_main()

# Carregar no rascunho (concatenar com segurança)
sel_text, sel_upl, load_to_draft = render_prompts_selector(prompts_bank=prompts_md)
if load_to_draft:
    base = (st.session_state.get("draft_prompt") or "").strip()
    parts = [base]
    if sel_text: parts.append(str(sel_text).strip())
    if sel_upl:  parts.append(str(sel_upl).strip())
    st.session_state["draft_prompt"] = "\n".join([p for p in parts if p]).strip()
    st.rerun()

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
    user_input = (user_text or st.session_state.get("draft_prompt") or "").strip()
    if not user_input:
        st.warning("Escreva algo no 'Conteúdo do prompt' ou em 'Texto de análise (para Sphera)' antes de enviar.")
        st.stop()

    # 1) Recuperação Sphera
    df_base = filter_sphera(df_sph, locations, substr, years)
    hits = topk_similar(user_input, df_base, E_sph, k_sph, thr_sph)
    loc_col_effective = get_sphera_location_col(df_base or df_sph)

    with st.expander("🔧 Diagnóstico RAG", expanded=False):
        st.write({
            "len(df_sph)": 0 if df_sph is None else len(df_sph),
            "len(df_base)": 0 if (df_base is None) else len(df_base),
            "E_sph.shape": None if E_sph is None else tuple(E_sph.shape),
            "hits": len(hits),
            "k_sph": k_sph, "thr_sph": float(thr_sph),
        })

    # 2) Agregação dicionários
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
    dic_res, debug_raw = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        per_event_thr=per_event_thr, support_min=support_min, agg_mode=agg_mode,
        thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
        top_ws=top_ws, top_prec=top_prec, top_cp=top_cp
    )

    # 3) Tabela hits
    st.subheader(f"Eventos do Sphera (Top-{min(k_sph, len(hits))})")
    df_hits = hits_dataframe(hits, loc_col_effective)
    st.dataframe(df_hits, use_container_width=True, hide_index=True)

    # 4) Depuração
    from ui.tables import show_debug_raw
    show_debug_raw(debug_raw)

    # 5) Contexto ao LLM (texto — o modelo NÃO “vê” embeddings, vê o contexto)
    ctx_lines = [
        datasets_ctx,  # seus arquivos .md globais continuam injetados sempre
        build_sphera_context_md(hits, loc_col_effective),           # já existia
        build_gosee_context_md(hits_gosee),                         # novo (se você estiver recuperando GoSee)
        build_investigation_context_md(hits_inc),                   # novo (se você estiver recuperando relatórios)
        build_dic_matches_md(dic_res),                              # WS/Precursores/CP agregados somente sobre Sphera
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])

    messages = [
        {"role":"system", "content":"Você é o SAFETY • CHAT. Baseie-se no contexto fornecido e nas regras da organização para ESO."},
        {"role":"user", "content": user_input},
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
