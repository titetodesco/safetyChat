from __future__ import annotations
import streamlit as st
import pandas as pd

# --- Config & Core ------------------------------------------------------------
import config as cfg

from core.data_loader import (
    load_sphera,
    load_datasets_context,
    load_prompts_md,          # mantido por compatibilidade (não usado aqui)
    load_dicts,
)

from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.context_builder import (
    hits_dataframe,
    build_dic_matches_md,
    build_sphera_context_md,
)
from core.dictionaries import aggregate_dict_matches_over_hits

# Se não tiver estes serviços, comente as duas linhas abaixo.
# from services.upload_extract import extract_any
from services.llm_client import chat

# --- Página -------------------------------------------------------------------
st.set_page_config(page_title="SAFETY • CHAT", layout="wide")
st.title("SAFETY • CHAT")

# --------------------- Estado base (sempre ANTES de widgets) ------------------
ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])

# --------------------- Carregamentos ------------------------------------------
# Contextos fixos (MD). Mantidos; o texto aparece na coluna direita.
datasets_ctx = load_datasets_context(cfg.DATASETS_CONTEXT_PATH)
_ = load_prompts_md(cfg.PROMPTS_MD_PATH)  # compatibilidade

# Dados/embeddings Sphera
df_sph, E_sph = load_sphera()  # confia no config.py atual

# --------------------- SIDEBAR (parâmetros) -----------------------------------
with st.sidebar:
    st.header("Recuperação – Sphera")
    k_sph = st.slider("Top-K Sphera", 5, 100, 20, step=5, key="sb_topk_sph")
    thr_sph = st.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01, key="sb_thr_sph")
    years = st.slider("Últimos N anos", 0, 10, 3, 1, key="sb_years")

    st.header("Filtros avançados – Sphera")
    substr = st.text_input("Description contém (substring)", value="", key="sb_substr")
    # opções de localização (se existir a coluna)
    loc_col_detected = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    loc_opts = sorted([x for x in df_sph[loc_col_detected].dropna().unique().tolist()]) if loc_col_detected else []
    locations = st.multiselect("Location (coluna: LOCATION)", options=loc_opts, default=[], key="sb_locations")

    st.header("Agregação sobre eventos recuperados (Sphera)")
    agg_mode = st.selectbox("Agregação", options=["count", "sum"], index=0, key="sb_agg_mode")
    per_event_thr = st.slider("Min. itens por evento (p/contar)", 0, 20, 0, 1, key="sb_per_event_thr")
    support_min = st.slider("Suporte mínimo (nº eventos)", 1, 50, 1, 1, key="sb_support_min")
    # limiares dos dicionários
    thr_ws = st.slider("Limiar WS", 0.0, 1.0, 0.30, 0.01, key="sb_thr_ws")
    thr_prec = st.slider("Limiar Precursores", 0.0, 1.0, 0.30, 0.01, key="sb_thr_prec")
    thr_cp = st.slider("Limiar CP", 0.0, 1.0, 0.30, 0.01, key="sb_thr_cp")
    # top-N de cada dicionário
    top_ws = st.slider("Top-N WS", 1, 50, 10, 1, key="sb_top_ws")
    top_prec = st.slider("Top-N Precursores", 1, 50, 10, 1, key="sb_top_prec")
    top_cp = st.slider("Top-N CP", 1, 50, 10, 1, key="sb_top_cp")

# --------------------- Colunas principais -------------------------------------
col_main, col_ctx = st.columns([3, 1])

with col_main:
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
        "Anexe .txt / .md / .csv",
        type=["txt", "md", "csv", "pdf", "docx", "xlsx"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if upl is not None:
        # Se tiver seu extractor, descomente a linha abaixo e remova a fallback simples:
        # uploaded_text = extract_any(upl)
        try:
            uploaded_text = upl.read().decode("utf-8", errors="ignore")
        except Exception:
            uploaded_text = ""
        if uploaded_text.strip():
            ss.upld_texts.append(uploaded_text)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        go_btn = st.button("Enviar para o chat", type="primary")
    with c2:
        if st.button("Limpar rascunho"):
            ss["draft_prompt"] = ""
            ss["analysis_text"] = ""
            ss["upld_texts"] = []
            st.rerun()
    with c3:
        if st.button("Limpar chat"):
            ss["chat"] = []
            st.rerun()

with col_ctx:
    st.subheader("Contexto fixo (datasets)")
    st.caption(cfg.DATASETS_CONTEXT_PATH.name)
    st.text_area("", datasets_ctx or "", height=380, label_visibility="collapsed")

# --------------------- Execução ------------------------------------------------
if go_btn:
    # 0) Compose user input
    user_parts = [draft, analysis] + (ss.upld_texts or [])
    user_input = "\n\n".join([p for p in user_parts if p]).strip()

    # 1) Recuperação no Sphera
    loc_col = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    hits = []
    if isinstance(df_base, pd.DataFrame) and not df_base.empty and E_sph is not None and user_input:
        hits = topk_similar(
            user_input,
            df_base,
            E_sph,
            topk=int(k_sph),
            min_sim=float(thr_sph),
        )

    st.subheader(f"Eventos do Sphera (Top-{min(int(k_sph), len(hits))})")
    if hits:
        st.dataframe(
            hits_dataframe(hits, loc_col),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Nenhum evento recuperado. Ajuste o texto/limiar/Top-K.")

    # 2) Agregação de dicionários (só se houver hits)
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

    # 3) Compõe contexto para o LLM
    ctx_lines = [
        datasets_ctx,
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])

    messages = [
        {"role": "system", "content": "Você é o SAFETY • CHAT. Baseie-se no contexto fornecido e nas regras da organização para ESO."},
        {"role": "user", "content": user_input},
        {"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
    ]

    try:
        res = chat(messages, stream=False)
        content = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        content = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(content)
    ss.chat.append({"role": "assistant", "content": content})

# ------------- Histórico ------------------------------------------------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss.chat[-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
