# app.py
from __future__ import annotations

# --- Path fix (Streamlit Cloud imports) ---------------------------------------
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

# --- Config & Core ------------------------------------------------------------
import config as cfg

from core.data_loader import (
    load_sphera,
    load_datasets_context,   # carregamos mas NÃO exibimos
    load_prompts_md,         # compatibilidade (não usado)
    load_dicts,
)

from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.context_builder import (
    hits_dataframe,
    build_dic_matches_md,
    build_sphera_context_md,
)
from core.dictionaries import aggregate_dict_matches_over_hits

from services.upload_extract import extract_any
from services.llm_client import chat


# --- Callbacks (antes dos widgets com keys) -----------------------------------
def clear_draft():
    st.session_state["draft_prompt"] = ""
    st.session_state["analysis_text"] = ""
    st.session_state["upld_texts"] = []


def clear_chat():
    st.session_state["chat"] = []
    for k in ["messages", "history", "chat_messages", "last_reply", "last_ctx", "last_hits"]:
        if k in st.session_state:
            del st.session_state[k]


# --- Página -------------------------------------------------------------------
st.set_page_config(page_title="SAFETY • CHAT", layout="wide")
st.title("SAFETY • CHAT")

# --------------------- Estado base (sempre ANTES de widgets) ------------------
ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])

# --------------------- Carregamentos silenciosos ------------------------------
_ = load_datasets_context(cfg.DATASETS_CONTEXT_PATH)   # NÃO renderiza
_ = load_prompts_md(cfg.PROMPTS_MD_PATH)               # NÃO renderiza

df_sph, E_sph = load_sphera()

# --------------------- SIDEBAR (parâmetros) -----------------------------------
with st.sidebar:
    st.header("Recuperação – Sphera")
    k_sph   = st.slider("Top-K Sphera", 5, 100, 20, step=5, key="sb_topk_sph")
    thr_sph = st.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01, key="sb_thr_sph")
    years   = st.slider("Últimos N anos", 0, 10, 3, 1, key="sb_years")

    st.header("Filtros avançados – Sphera")
    substr = st.text_input("Description contém (substring)", value="", key="sb_substr")

    loc_col_detected = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    loc_opts = (
        sorted(df_sph[loc_col_detected].dropna().unique().tolist())
        if (isinstance(df_sph, pd.DataFrame) and loc_col_detected in df_sph.columns)
        else []
    )
    locations = st.multiselect("Location", options=loc_opts, default=[], key="sb_locations")

    st.header("Agregação sobre eventos recuperados (Sphera)")
    agg_mode = st.selectbox("Agregação", options=["max", "mean"], index=0, key="sb_agg_mode")
    per_event_thr = st.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01, key="sb_per_event_thr")
    support_min = st.slider("Suporte mínimo (nº eventos)", 1, 50, 2, 1, key="sb_support_min")

    st.markdown("---")
    thr_ws   = st.slider("Limiar WS", 0.0, 1.0, 0.30, 0.01, key="sb_thr_ws")
    thr_prec = st.slider("Limiar Precursores", 0.0, 1.0, 0.30, 0.01, key="sb_thr_prec")
    thr_cp   = st.slider("Limiar CP", 0.0, 1.0, 0.30, 0.01, key="sb_thr_cp")

    top_ws   = st.slider("Top-N WS", 1, 50, 10, 1, key="sb_top_ws")
    top_prec = st.slider("Top-N Precursores", 1, 50, 10, 1, key="sb_top_prec")
    top_cp   = st.slider("Top-N CP", 1, 50, 10, 1, key="sb_top_cp")

# --------------------- Área principal -----------------------------------------
st.subheader("Conteúdo do prompt")
draft = st.text_area(
    "Digite ou carregue um modelo de prompt…",
    key="draft_prompt", height=220, label_visibility="collapsed",
)

st.subheader("Texto de análise (para Sphera)")
analysis = st.text_area(
    "Cole aqui a descrição/evento a analisar…",
    key="analysis_text", height=220, label_visibility="collapsed",
)

st.subheader("Anexar arquivo (opcional)")
upl = st.file_uploader(
    "Anexe .txt / .md / .csv / .pdf / .docx / .xlsx",
    type=["txt", "md", "csv", "pdf", "docx", "xlsx"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)
if upl is not None:
    uploaded_text = extract_any(upl)
    if uploaded_text.strip():
        ss.upld_texts.append(uploaded_text)
        st.success(f"Upload recebido: {upl.name}")
    else:
        st.warning(f"Não foi possível extrair texto de {upl.name}.")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    go_btn = st.button("Enviar para o chat", type="primary")
with c2:
    st.button("Limpar rascunho", on_click=clear_draft)
with c3:
    st.button("Limpar chat", on_click=clear_chat)

# --------------------- Execução ------------------------------------------------
if go_btn:
    # ✅ Retrieval deve usar o texto do incidente (analysis) para não "poluir" o embedding
    query_for_retrieval = (analysis or "").strip()

    # Entrada do usuário para o chat pode continuar sendo "tudo"
    user_parts = [draft, analysis] + (ss.upld_texts or [])
    user_input = "\n\n".join([p for p in user_parts if p]).strip()

    # 1) Recuperação Sphera
    loc_col = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    hits = []
    if (
        isinstance(df_base, pd.DataFrame)
        and not df_base.empty
        and E_sph is not None
        and query_for_retrieval
    ):
        # ✅ ALINHAMENTO: filtra embeddings com base no _rowid
        if "_rowid" not in df_base.columns:
            raise KeyError(
                "[Sphera] Coluna '_rowid' não encontrada no df_base. "
                "Garanta que load_sphera() cria _rowid para alinhamento DF<->embeddings."
            )

        rowids = df_base["_rowid"].to_numpy()
        E_base = E_sph[rowids]

        hits = topk_similar(
            query_for_retrieval,   # ✅ usa só o incidente
            df_base.reset_index(drop=True),
            E_base,
            topk=int(k_sph),
            min_sim=float(thr_sph),
        )

    st.subheader(f"Eventos do Sphera (Top-{min(int(k_sph), len(hits))})")
    if hits:
        st.dataframe(hits_dataframe(hits, loc_col), width="stretch", hide_index=True)
    else:
        st.info("Nenhum evento recuperado. Ajuste o texto/limiar/Top-K.")

    # 2) Agregação dicionários
    dic_res, debug_raw = {}, {}
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits,
            E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
            per_event_thr=float(per_event_thr),
            support_min=int(support_min),
            agg_mode=str(agg_mode),
            thr_ws=float(thr_ws), thr_prec=float(thr_prec), thr_cp=float(thr_cp),
            top_ws=int(top_ws), top_prec=int(top_prec), top_cp=int(top_cp),
        )

    # 3) Contexto para o LLM
    ctx_full = "\n".join([
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ])

    # ✅ CONTEXTO COMO SYSTEM: reduz duplicação / melhora obediência
    messages = [
        {
            "role": "system",
            "content": (
                "Você é o SAFETY • CHAT. Use o CONTEXTO fornecido apenas como suporte factual. "
                "Não repita blocos ou seções. Não duplique recomendações."
            ),
        },
        {"role": "system", "content": "CONTEXTO:\n" + ctx_full},
        {"role": "user", "content": user_input},
    ]

    try:
        res = chat(messages, stream=False)
        reply = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        reply = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(reply)
    ss.chat.append({"role": "assistant", "content": reply})

# ------------- Histórico (últimas 10) -----------------------------------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss.chat[-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
