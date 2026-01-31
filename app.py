from __future__ import annotations

import os
import importlib
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd

# ---------- Config (import robusto) ----------
try:
    cfg = importlib.import_module("config")
except Exception as e:
    st.error(f"[FATAL] Falha ao importar config.py: {e}")
    raise

# ---------- Core loaders ----------
from core.data_loader import (
    load_sphera,
    load_gosee,
    load_incidents,
    load_datasets_context,
    load_prompts_md,
    load_dicts,
)

# ---------- Core RAG helpers ----------
# ATENÇÃO: estas funções DEVEM existir em core/sphera.py
#   get_sphera_location_col(df), filter_sphera(df, locations, substr, years), topk_similar(query, df, E, topk, min_sim)
from core.sphera import get_sphera_location_col, filter_sphera, topk_similar

# ---------- Context builders ----------
from core.context_builder import (
    hits_dataframe,
    build_dic_matches_md,
    build_sphera_context_md,
)

# ---------- Dicionários / agregação ----------
from core.dictionaries import aggregate_dict_matches_over_hits

# ---------- Serviços LLM e upload ----------
try:
    from services.upload_extract import extract_any  # se existir
except Exception:
    # Fallback mínimo para não quebrar o app se o módulo não estiver no repo
    def extract_any(f) -> str:
        try:
            name = getattr(f, "name", "").lower()
            data = f.read()
            if isinstance(data, bytes):
                try:
                    data = data.decode("utf-8", errors="ignore")
                except Exception:
                    data = ""
            # suporte simples a txt/md/csv
            if name.endswith((".txt", ".md", ".csv")):
                return str(data)
            return ""
        except Exception:
            return ""

try:
    from services.llm_client import chat
except Exception as e:
    chat = None  # não bloqueia a UI; apenas desabilita a chamada ao modelo


# ---------- Sidebar UI (sem assistente de prompt) ----------
from ui.sidebar import (
    render_retrieval_controls,
    render_advanced_filters,
    render_aggregation_controls,
    render_util_buttons,
)

# ---------- Página ----------
st.set_page_config(page_title="SAFETY • CHAT", layout="wide")
st.title("SAFETY • CHAT")

# ---------- Estado base (SEM debug solto no topo) ----------
ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])

# ---------- Carregamentos base (dataset + contexto + prompts_md) ----------
# (prompts_md não é obrigatório para rodar, mas carregamos por compatibilidade)
datasets_ctx = load_datasets_context(cfg.DATASETS_CONTEXT_PATH)
_ = load_prompts_md(cfg.PROMPTS_MD_PATH)

df_sph, E_sph = load_sphera()             # Sphera (parquet + npz)
df_gosee, E_gosee = load_gosee()          # GoSee   (parquet + npz) — se você ainda não usa, mantém carregado
df_inc, E_inc = load_incidents()          # Incidents/History (parquet/jsonl + npz)

# ---------- Entrada principal ----------
colL, colR = st.columns([2, 1], gap="large")

with colL:
    st.subheader("Prompt (texto do caso atual)")
    user_text = st.text_area(
        "Conteúdo do prompt",
        key="draft_prompt",
        height=220,
        placeholder="Descreva o evento/situação aqui...",
    )

    st.subheader("Upload de arquivo (opcional)")
    upl_files = st.file_uploader(
        "Anexe .txt / .md / .csv",
        type=["txt", "md", "csv"],
        accept_multiple_files=True,
    )
    new_texts: List[str] = []
    if upl_files:
        for f in upl_files:
            try:
                new_texts.append(extract_any(f) or "")
            except Exception:
                new_texts.append("")
    if new_texts:
        ss["upld_texts"] = new_texts

    go_btn = st.button("Executar análise", type="primary")

with colR:
    st.subheader("Contexto fixo (datasets)")
    st.text_area(
        "datasets_context.md",
        value=datasets_ctx or "",
        height=240,
        key="datasets_ctx_view",
        disabled=True,
    )

# ---------- Sidebar (parâmetros) ----------
with st.sidebar:
    k_sph, thr_sph, years = render_retrieval_controls()
    locations, substr, loc_col_override, loc_opts = render_advanced_filters(df_sph)
    (
        agg_mode,
        per_event_thr,
        support_min,
        thr_ws,
        thr_prec,
        thr_cp,
        top_ws,
        top_prec,
        top_cp,
    ) = render_aggregation_controls()
    clear_upl, clear_chat_btn = render_util_buttons()

if clear_upl:
    ss["upld_texts"] = []
if clear_chat_btn:
    ss["chat"] = []

# ---------- Execução (quando clicar em Executar análise) ----------
if go_btn:
    # 1) Texto de entrada (prompt + uploads)
    user_input_parts = [user_text]
    if ss.get("upld_texts"):
        user_input_parts.extend([t for t in ss["upld_texts"] if t])
    user_input = "\n\n".join([p for p in user_input_parts if p]).strip()

    # 2) Recuperação Sphera
    loc_col = loc_col_override or (get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None)
    df_base = filter_sphera(df_sph, locations, substr, years)

    hits: List[Tuple[str, float, pd.Series]] = []
    if isinstance(df_base, pd.DataFrame) and not df_base.empty and E_sph is not None and user_input:
        hits = topk_similar(
            query_text=user_input,
            df=df_base,
            E=E_sph,
            topk=k_sph,
            min_sim=thr_sph,     # o slider da barra lateral
        )

    st.subheader(f"Eventos do Sphera (Top-{min(int(k_sph), len(hits))})")
    if hits:
        st.dataframe(
            hits_dataframe(hits, loc_col),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Nenhum evento recuperado. Ajuste o texto/limiar/Top-K ou filtros.")

    # 3) Agregação dicionários (só se houver hits)
    dic_res, debug_raw = {}, {}
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits,
            E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
            per_event_thr=per_event_thr,
            support_min=support_min,
            agg_mode=agg_mode,
            thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
            top_ws=top_ws, top_prec=top_prec, top_cp=top_cp,
        )

    # 4) Monta contexto para LLM
    ctx_lines = [
        datasets_ctx or "",
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c]).strip()

    # 5) Chamada ao modelo (se disponível)
    assistant_content = ""
    if chat is None:
        assistant_content = "⚠️ Serviço de LLM indisponível neste momento (services.llm_client)."
    else:
        messages = [
            {"role": "system", "content": "Você é o SAFETY • CHAT. Baseie-se no contexto fornecido e nas regras da organização para ESO."},
            {"role": "user",   "content": user_input or ""},
            {"role": "user",   "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
        ]
        try:
            res = chat(messages, stream=False)
            assistant_content = res.get("message", {}).get("content", "") or "(sem conteúdo)"
        except Exception as e:
            assistant_content = f"Falha ao consultar o modelo: {e}"

    # 6) Apresenta resposta
    with st.chat_message("assistant"):
        st.markdown(assistant_content)

    ss["chat"].append({"role": "assistant", "content": assistant_content})

# ---------- Histórico ----------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss["chat"][-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
