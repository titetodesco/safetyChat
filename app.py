# app.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

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


# ---------------- Helpers ----------------
def _ensure_eventid_column(df: pd.DataFrame) -> pd.DataFrame:
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
            return df.rename(columns={c: "EventID"})

    df = df.copy()
    if "_rowid" in df.columns:
        df["EventID"] = df["_rowid"].apply(lambda x: f"ROW_{x}")
    else:
        df["EventID"] = [f"ROW_{i}" for i in range(len(df))]
    return df


def _safe_event_ids_from_hits(hits) -> list[str]:
    ids: list[str] = []
    for evid, _, _ in hits:
        s = str(evid).strip() if evid is not None else ""
        if s and s not in ids:
            ids.append(s)
    return ids


# ---------------- Callbacks ----------------
def clear_draft():
    st.session_state["draft_prompt"] = ""
    st.session_state["analysis_text"] = ""
    st.session_state["upld_texts"] = []


def clear_chat():
    st.session_state["chat"] = []
    for k in ["messages", "history", "chat_messages", "last_reply", "last_ctx", "last_hits"]:
        if k in st.session_state:
            del st.session_state[k]


# ---------------- Page ----------------
st.set_page_config(page_title="SAFETY • CHAT", layout="wide")
st.title("SAFETY • CHAT")

ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])

# carregamentos silenciosos
_ = load_datasets_context(cfg.DATASETS_CONTEXT_PATH)
_ = load_prompts_md(cfg.PROMPTS_MD_PATH)

df_sph, E_sph = load_sphera()
df_sph = _ensure_eventid_column(df_sph)

# ---------------- Sidebar ----------------
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
    support_min = st.slider("Suporte mínimo (nº eventos)", 1, 50, 20, 1, key="sb_support_min")

    st.markdown("---")
    thr_ws = st.slider("Limiar WS", 0.0, 1.0, 0.50, 0.01, key="sb_thr_ws")
    thr_prec = st.slider("Limiar Precursores", 0.0, 1.0, 0.40, 0.01, key="sb_thr_prec")
    thr_cp = st.slider("Limiar CP", 0.0, 1.0, 0.30, 0.01, key="sb_thr_cp")

    top_ws = st.slider("Top-N WS", 1, 50, 10, 1, key="sb_top_ws")
    top_prec = st.slider("Top-N Precursores", 1, 50, 10, 1, key="sb_top_prec")
    top_cp = st.slider("Top-N CP", 1, 50, 10, 1, key="sb_top_cp")

# ---------------- Main ----------------
st.subheader("Conteúdo do prompt")
draft = st.text_area("Prompt", key="draft_prompt", height=220, label_visibility="collapsed")

st.subheader("Texto de análise (para Sphera)")
analysis = st.text_area("Análise", key="analysis_text", height=220, label_visibility="collapsed")

st.subheader("Anexar arquivo (opcional)")
upl = st.file_uploader(
    "Upload",
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

# ---------------- Run ----------------
if go_btn:
    # ✅ Retrieval usa somente o incidente
    query_for_retrieval = (analysis or "").strip()

    # input do chat pode ter tudo
    user_parts = [draft, analysis] + (ss.upld_texts or [])
    user_input = "\n\n".join([p for p in user_parts if p]).strip()

    # sempre inicializa (evita NameError)
    hits = []
    dic_res, debug_raw = {"WS": [], "Precursores": [], "CP": []}, {}
    ws_matches, prec_matches, cp_matches = [], [], []

    # 1) filtra df
    loc_col = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    # 2) topk
    if isinstance(df_base, pd.DataFrame) and not df_base.empty and E_sph is not None and query_for_retrieval:
        if "_rowid" not in df_base.columns:
            st.error("Sphera sem coluna _rowid. Verifique load_sphera() em core/data_loader.py.")
        else:
            rowids = df_base["_rowid"].to_numpy()
            E_base = E_sph[rowids]
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
        df_hits = hits_dataframe(hits, loc_col)
        if "EventID" not in df_hits.columns:
            df_hits.insert(0, "EventID", [h[0] for h in hits])
        st.dataframe(df_hits, width="stretch", hide_index=True)
    else:
        st.info("Nenhum evento recuperado. Ajuste texto/limiar/Top-K.")

    # 3) dicionários (✅ agora com os 6 argumentos obrigatórios)
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()

        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits,
            E_ws, L_ws,
            E_prec, L_prec,
            E_cp, L_cp,
            per_event_thr=float(per_event_thr),
            support_min=int(support_min),
            agg_mode=str(agg_mode),
            thr_ws=float(thr_ws),
            thr_prec=float(thr_prec),
            thr_cp=float(thr_cp),
            top_ws=int(top_ws),
            top_prec=int(top_prec),
            top_cp=int(top_cp),
        )

        ws_matches = dic_res.get("WS", []) if isinstance(dic_res, dict) else []
        prec_matches = dic_res.get("Precursores", []) if isinstance(dic_res, dict) else []
        cp_matches = dic_res.get("CP", []) if isinstance(dic_res, dict) else []

    # Mostra WS determinístico
    st.subheader("Weak Signals (calculado por embeddings, sem LLM)")
    if ws_matches:
        st.dataframe(pd.DataFrame(ws_matches, columns=["Termo", "Score"]), width="stretch", hide_index=True)
    else:
        st.info("Nenhum WS acima do limiar atual.")

    # 4) contexto e guardrails
    allowed_event_ids = _safe_event_ids_from_hits(hits)

    ctx_full = "\n".join([
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ])

    ws_list = [str(t[0]).strip() for t in ws_matches]
    prec_list = [str(t[0]).strip() for t in prec_matches]
    cp_list = [str(t[0]).strip() for t in cp_matches]

    ws_block = "WS_MATCHES (autoritativo; NÃO invente IDs/códigos):\n" + (
        "\n".join([f"- {t}" for t in ws_list]) if ws_list else "- (nenhum)\n"
    )
    prec_block = "PRECURSORES_MATCHES:\n" + (
        "\n".join([f"- {t}" for t in prec_list]) if prec_list else "- (nenhum)\n"
    )
    cp_block = "CP_MATCHES:\n" + (
        "\n".join([f"- {t}" for t in cp_list]) if cp_list else "- (nenhum)\n"
    )

    guardrails = (
        "REGRAS OBRIGATÓRIAS:\n"
        "1) NÃO invente WS/Precursores/CP. Use APENAS os termos listados em *_MATCHES.\n"
        "2) NÃO use 'WS ID', 'WS code', 'WS1/WS2' ou numeração. O dicionário não tem IDs.\n"
        "3) Ao citar eventos, use APENAS EventIDs desta lista (não invente): "
        f"{', '.join(allowed_event_ids) if allowed_event_ids else '(nenhum)'}\n"
        "4) Se não houver termos acima do limiar, diga explicitamente que não encontrou.\n"
    )

    messages = [
        {"role": "system", "content": "Você é o SAFETY • CHAT. Seja preciso e não alucine."},
        {"role": "system", "content": guardrails},
        {"role": "system", "content": ws_block + "\n\n" + prec_block + "\n\n" + cp_block},
        {"role": "system", "content": "CONTEXTO (eventos recuperados do Sphera):\n" + ctx_full},
        {"role": "user", "content": user_input},
    ]

    try:
        res = chat(messages, stream=False)
        reply = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        reply = f"Falha ao consultar o modelo: {e}"

    # bloqueio simples se insistir em IDs
    rl = reply.lower()
    if ("ws id" in rl) or ("ws code" in rl) or ("ws1" in rl) or ("ws2" in rl) or ("ws3" in rl):
        reply = (
            "⚠️ A resposta do modelo foi bloqueada porque tentou inventar códigos/IDs de WS.\n\n"
            "Use a tabela 'Weak Signals (calculado por embeddings, sem LLM)' como fonte.\n"
        )

    with st.chat_message("assistant"):
        st.markdown(reply)
    ss.chat.append({"role": "assistant", "content": reply})

# ---------------- History ----------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss.chat[-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
