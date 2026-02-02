# app.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

import config as cfg

from core.data_loader import (
    load_sphera,
    load_datasets_context,
    load_prompts_md,
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


# --- Helpers -----------------------------------------------------------------
def _ensure_eventid_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que existe uma coluna EventID (com o ID real do evento).
    Se existir uma coluna equivalente, renomeia para EventID.
    """
    if df is None or df.empty:
        return df

    if "EventID" in df.columns:
        return df

    candidates = [
        "EVENTID", "EVENT_ID", "Event ID", "EVENT ID", "ID", "Id", "id",
        "EventId", "eventid", "event_id",
    ]
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: "EventID"})
            return df

    # se não existir nenhum, cria EventID com base no _rowid (não ideal, mas explícito)
    if "_rowid" in df.columns:
        df["EventID"] = df["_rowid"].apply(lambda x: f"ROW_{x}")
    else:
        df["EventID"] = [f"ROW_{i}" for i in range(len(df))]
    return df


def _safe_event_ids_from_hits(hits) -> list[str]:
    ids = []
    for evid, _, _ in hits:
        if evid is None:
            continue
        s = str(evid).strip()
        if s and s not in ids:
            ids.append(s)
    return ids


# --- Callbacks ---------------------------------------------------------------
def clear_draft():
    st.session_state["draft_prompt"] = ""
    st.session_state["analysis_text"] = ""
    st.session_state["upld_texts"] = []


def clear_chat():
    st.session_state["chat"] = []
    for k in ["messages", "history", "chat_messages", "last_reply", "last_ctx", "last_hits"]:
        if k in st.session_state:
            del st.session_state[k]


# --- Page --------------------------------------------------------------------
st.set_page_config(page_title="SAFETY • CHAT", layout="wide")
st.title("SAFETY • CHAT")

ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])

_ = load_datasets_context(cfg.DATASETS_CONTEXT_PATH)
_ = load_prompts_md(cfg.PROMPTS_MD_PATH)

df_sph, E_sph = load_sphera()
df_sph = _ensure_eventid_column(df_sph)  # ✅ garante EventID real/consistente

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Recuperação – Sphera")
    k_sph = st.slider("Top-K Sphera", 5, 100, 20, step=5, key="sb_topk_sph")
    thr_sph = st.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01, key="sb_thr_sph")
    years = st.slider("Últimos N anos", 0, 10, 3, 1, key="sb_years")

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
    thr_ws = st.slider("Limiar WS", 0.0, 1.0, 0.30, 0.01, key="sb_thr_ws")
    thr_prec = st.slider("Limiar Precursores", 0.0, 1.0, 0.30, 0.01, key="sb_thr_prec")
    thr_cp = st.slider("Limiar CP", 0.0, 1.0, 0.30, 0.01, key="sb_thr_cp")

    top_ws = st.slider("Top-N WS", 1, 50, 10, 1, key="sb_top_ws")
    top_prec = st.slider("Top-N Precursores", 1, 50, 10, 1, key="sb_top_prec")
    top_cp = st.slider("Top-N CP", 1, 50, 10, 1, key="sb_top_cp")

# --- Main --------------------------------------------------------------------
st.subheader("Conteúdo do prompt")
draft = st.text_area(
    "Digite ou carregue um modelo de prompt…",
    key="draft_prompt",
    height=220,
    label_visibility="collapsed",
)

st.subheader("Texto de análise (para Sphera)")
analysis = st.text_area(
    "Cole aqui a descrição/evento a analisar…",
    key="analysis_text",
    height=220,
    label_visibility="collapsed",
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

# --- Run ---------------------------------------------------------------------
if go_btn:
    # ✅ retrieval: usa somente o texto do incidente (analysis)
    query_for_retrieval = (analysis or "").strip()

    # chat input pode ter tudo (prompt + analysis + uploads)
    user_parts = [draft, analysis] + (ss.upld_texts or [])
    user_input = "\n\n".join([p for p in user_parts if p]).strip()

    # 1) Filtra DF
    loc_col = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    hits = []
    if (
        isinstance(df_base, pd.DataFrame)
        and not df_base.empty
        and E_sph is not None
        and query_for_retrieval
    ):
        # ✅ alinhamento embeddings com DF filtrado via _rowid
        if "_rowid" not in df_base.columns:
            raise KeyError(
                "[Sphera] Coluna '_rowid' não encontrada. Garanta que load_sphera() cria _rowid."
            )

        rowids = df_base["_rowid"].to_numpy()
        E_base = E_sph[rowids]

        # reset index para garantir iloc consistente
        df_base2 = df_base.reset_index(drop=True)

        hits = topk_similar(
            query_for_retrieval,
            df_base2,
            E_base,
            topk=int(k_sph),
            min_sim=float(thr_sph),
        )

    st.subheader(f"Eventos do Sphera (Top-{min(int(k_sph), len(hits))})")
    if hits:
        # garante que tabela mostre EventID (não index)
        df_hits = hits_dataframe(hits, loc_col)
        if "EventID" not in df_hits.columns:
            # tenta reconstruir EventID a partir do tuple (evid)
            df_hits.insert(0, "EventID", [h[0] for h in hits])
        st.dataframe(df_hits, width="stretch", hide_index=True)
    else:
        st.info("Nenhum evento recuperado. Ajuste texto/limiar/Top-K.")

    # 2) Agregação dicionários (WS/Prec/CP) — calculado por embeddings (não pelo LLM)
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

    # 3) Contexto e guardrails anti-hallucination
    allowed_event_ids = _safe_event_ids_from_hits(hits)

    ctx_full = "\n".join([
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ])

    guardrails = (
        "REGRAS IMPORTANTES:\n"
        "1) Weak Signals (WS), Precursores e CP DEVEM vir APENAS do CONTEXTO (dicionários). NÃO invente WS.\n"
        "2) Ao citar eventos, use APENAS EventIDs existentes nesta lista. NÃO invente EventIDs.\n"
        f"EventIDs permitidos: {', '.join(allowed_event_ids) if allowed_event_ids else '(nenhum)'}\n"
        "3) Se o CONTEXTO não trouxer WS/Prec/CP suficientes acima do limiar, diga explicitamente que não encontrou.\n"
    )

    messages = [
        {"role": "system", "content": "Você é o SAFETY • CHAT. Seja preciso e não alucine."},
        {"role": "system", "content": guardrails},
        {"role": "system", "content": "CONTEXTO (use como fonte):\n" + ctx_full},
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

# --- History -----------------------------------------------------------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss.chat[-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
