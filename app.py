from __future__ import annotations

import streamlit as st
import pandas as pd

# --- imports do projeto
from config import DATASETS_CONTEXT_PATH, PROMPTS_MD_PATH
from core.data_loader import (
    load_sphera, load_prompts_md, load_datasets_context, load_dicts,
)
from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.context_builder import hits_dataframe, build_dic_matches_md, build_sphera_context_md
from core.dictionaries import aggregate_dict_matches_over_hits
from services.upload_extract import extract_any
from services.llm_client import chat
from ui.sidebar import (
    render_prompts_selector,
    render_retrieval_controls,
    render_advanced_filters,
    render_aggregation_controls,
    render_util_buttons,
)

st.set_page_config(page_title="SAFETY • CHAT", layout="wide")

# --------------------- estado base (sempre antes dos widgets) ---------------------
if "draft_prompt" not in st.session_state:
    st.session_state["draft_prompt"] = ""
if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "upld_texts" not in st.session_state:
    st.session_state["upld_texts"] = []

# --------------------- carregamentos (dados, prompts, contexto) -------------------
datasets_ctx = load_datasets_context(DATASETS_CONTEXT_PATH) or ""
prompts_md   = load_prompts_md(PROMPTS_MD_PATH) or {"texto": [], "upload": []}

df_sph, E_sph = load_sphera()

# --------------------- SIDEBAR (parâmetros) --------------------------------------
with st.sidebar:
    st.caption("Versão refatorada • parâmetros RAG e utilitários")

sel_text, sel_upl, load_to_draft = render_prompts_selector(prompts_bank=prompts_md)

k_sph, thr_sph, years = render_retrieval_controls()
locations, substr, loc_col_guess, loc_options = render_advanced_filters(df_sph)
(agg_mode, per_event_thr, support_min, thr_ws, thr_prec, thr_cp,
 top_ws, top_prec, top_cp) = render_aggregation_controls()
clear_upl, clear_chat_btn = render_util_buttons()

# ações utilitárias
if clear_upl:
    st.session_state["upld_texts"] = []
    st.success("Uploads limpos.")
if clear_chat_btn:
    st.session_state["chat"] = []
    st.success("Chat limpo.")

# carregar prompt do assistente no rascunho (SEM quebrar o widget)
if load_to_draft:
    base = (st.session_state.get("draft_prompt") or "").strip()
    parts = [base]
    if sel_text:
        parts.append(sel_text.strip())
    if sel_upl:
        parts.append(sel_upl.strip())
    st.session_state["draft_prompt"] = "\n\n".join([p for p in parts if p])
    st.rerun()

# --------------------- UI principal (centro) --------------------------------------
st.title("SAFETY • CHAT")

col_main, = st.columns(1)

with col_main:
    draft = st.text_area("Conteúdo do prompt", key="draft_prompt", height=220)

    txt_for_sph = st.text_area("Texto de análise (para Sphera)",
                               placeholder="Descreva o cenário...", height=180)

    upl = st.file_uploader(
        "Anexar arquivo (opcional)",
        type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
        accept_multiple_files=True
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    go_btn = col_btn1.button("Enviar para o chat", use_container_width=True)
    clear_draft = col_btn2.button("Limpar rascunho", use_container_width=True)
    clear_chat = col_btn3.button("Limpar chat", use_container_width=True)

    if clear_draft:
        # zera antes de redesenhar o text_area
        st.session_state["draft_prompt"] = ""
        st.rerun()
    if clear_chat:
        st.session_state["chat"] = []
        st.rerun()

# extrair textos dos uploads (se houver)
if upl:
    extracted = []
    for f in upl:
        try:
            text = extract_any(f)
            if text:
                extracted.append(text)
        except Exception as e:
            st.warning(f"Falha ao extrair de {getattr(f,'name','(arquivo)')}: {e}")
    st.session_state["upld_texts"] = extracted

# --------------------- Execução (ao clicar) ---------------------------------------
if go_btn:
    user_input = (st.session_state.get("draft_prompt") or "").strip()
    # 1) Recuperação Sphera
    loc_col_eff = get_sphera_location_col(df_sph) if df_sph is not None else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    hits = []
    if (df_base is not None) and (E_sph is not None) and user_input:
        hits = topk_similar(user_input, df_base, E_sph, topk=k_sph, min_cos=thr_sph)

    # diagnóstico
    with st.expander("🛠️ Diagnóstico RAG", expanded=True):
        st.json({
            "len(df_sph)": len(df_sph) if isinstance(df_sph, pd.DataFrame) else 0,
            "len(df_base)": len(df_base) if isinstance(df_base, pd.DataFrame) else 0,
            "E_sph.shape": tuple(E_sph.shape) if E_sph is not None else None,
            "hits": len(hits) if hits else 0,
            "k_sph": k_sph,
            "thr_sph": thr_sph,
            "loc_col_effective": loc_col_eff,
        })

    st.subheader(f"Eventos do Sphera (Top-{min(k_sph, len(hits))})")
    if hits:
        st.dataframe(hits_dataframe(hits, loc_col_eff), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum evento recuperado. Ajuste o texto/limiar/Top-K.")

    # 2) Agregação dicionários (se houver hits)
    dic_res = {}
    debug_raw = {}
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
            per_event_thr=per_event_thr, support_min=support_min, agg_mode=agg_mode,
            thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
            top_ws=top_ws, top_prec=top_prec, top_cp=top_cp
        )

    # 3) Montar contexto para o LLM
    ctx_lines = [
        datasets_ctx,
        build_sphera_context_md(hits, loc_col_eff),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])

    # 4) Chamada ao modelo
    messages = [
        {"role": "system", "content": "Você é o SAFETY • CHAT. Responda como especialista em ESO usando o contexto abaixo sempre que útil."},
        {"role": "user", "content": (user_input or "")},
        {"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
    ]

    try:
        res = chat(messages, stream=False)
        answer = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        # Erro comum: 405 (endpoint errado). Explique de forma útil.
        msg = str(e)
        if "405" in msg and "ollama.com/api" in msg:
            answer = (
                "Falha ao consultar o modelo (HTTP 405). Verifique no seu services/llm_client.py "
                "se está usando **POST https://ollama.com/api/chat** (e não a raiz /api), "
                "com o header Authorization Bearer e JSON {model, messages}.")
        else:
            answer = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state["chat"].append({"role": "assistant", "content": answer})

# --------------------- Histórico -----------------------------------------------
if st.session_state.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in st.session_state["chat"][-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
