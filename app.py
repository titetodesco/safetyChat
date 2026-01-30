from __future__ import annotations
import streamlit as st
import pandas as pd

# --- Config & Core
from config import (
    DATASETS_CONTEXT_PATH, PROMPTS_MD_PATH,
)
from core.data_loader import (
    load_sphera, load_gosee, load_incidents,
    load_datasets_context, load_prompts_md, load_dicts,
)
from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.context_builder import (
    hits_dataframe, build_dic_matches_md, build_sphera_context_md
)
from core.dictionaries import aggregate_dict_matches_over_hits

# --- Serviços
from services.upload_extract import extract_any
from services.llm_client import chat

# --- Sidebar (sem Assistente de Prompt)
from ui.sidebar import (
    render_retrieval_controls,
    render_advanced_filters,
    render_aggregation_controls,
    render_util_buttons,
)

st.set_page_config(page_title="SAFETY • CHAT", layout="wide")


st.code({
  "len_df_sph": len(df_sph),
  "E_shape": tuple(E_sph.shape),
  "rowid_minmax": (int(df_sph["_rowid"].min()), int(df_sph["_rowid"].max()))
})





# --------------------- Estado base (sempre ANTES de widgets) ---------------------
ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])

# --------------------- Carregamento de contexto e datasets -----------------------
# Mantemos load_prompts_md para compatibilidade futura, mas não exibimos “assistente”.
_ = load_prompts_md(PROMPTS_MD_PATH)
datasets_ctx = load_datasets_context(DATASETS_CONTEXT_PATH) or ""

# Bases + Embeddings pré-computados
df_sph, E_sph = load_sphera()           # Sphera
df_gosee, E_gosee = load_gosee()        # GoSee (carregado, se quiser usar depois)
df_inc, E_inc = load_incidents()        # Histórico/Investigations (idem)

# --------------------- SIDEBAR ---------------------------------------------------
with st.sidebar:
    st.caption("Parâmetros RAG e utilitários")

k_sph, thr_sph, years = render_retrieval_controls()
locations, substr, loc_col_guess, loc_options = render_advanced_filters(df_sph)
(
    agg_mode, per_event_thr, support_min,
    thr_ws, thr_prec, thr_cp,
    top_ws, top_prec, top_cp
) = render_aggregation_controls()
clear_upl, clear_chat_btn = render_util_buttons()

if clear_upl:
    ss["upld_texts"] = []
    st.success("Uploads limpos.")
if clear_chat_btn:
    ss["chat"] = []
    st.success("Chat limpo.")

# --------------------- UI central ------------------------------------------------
st.title("SAFETY • CHAT")

draft = st.text_area(
    "Conteúdo do prompt",
    key="draft_prompt",
    height=220,
    placeholder="Descreva seu caso (o que aconteceu, onde, quando, condições, etc.)…",
)

analysis_text = st.text_area(
    "Texto de análise (para Sphera)",
    key="analysis_text",
    height=160,
    placeholder="(Opcional) Texto adicional para direcionar a busca de eventos no Sphera…",
)

upl_files = st.file_uploader(
    "Anexar arquivo (opcional)",
    type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
    accept_multiple_files=True,
    key="uploader_any",
)

col_go, col_clear_draft, col_clear_chat = st.columns([1, 1, 1])
go_btn      = col_go.button("Enviar para o chat", use_container_width=True, key="btn_go")
wipe_draft  = col_clear_draft.button("Limpar rascunho", use_container_width=True, key="btn_wipe_draft")
wipe_chat   = col_clear_chat.button("Limpar chat", use_container_width=True, key="btn_wipe_chat")

if wipe_draft:
    ss["draft_prompt"] = ""
    st.rerun()
if wipe_chat:
    ss["chat"] = []
    st.rerun()

# Extrai textos de uploads (se houver)
if upl_files:
    texts = []
    for f in upl_files:
        try:
            t = extract_any(f)
            if t:
                texts.append(t)
        except Exception as e:
            st.warning(f"Falha ao extrair de {getattr(f,'name','(arquivo)')}: {e}")
    ss["upld_texts"] = texts

# --------------------- Execução ao clicar ----------------------------------------
if go_btn:
    # Consolida entrada do usuário para rankear no Sphera
    user_input_parts = [
        (ss.get("draft_prompt") or "").strip(),
        (ss.get("analysis_text") or "").strip(),
        "\n\n".join(ss.get("upld_texts") or []),
    ]
    user_input = "\n\n".join([p for p in user_input_parts if p]).strip()

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

    # 2) Agregação WS/Precursores/CP (apenas se há hits)
    dic_res, debug_raw = {}, {}
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
            per_event_thr=per_event_thr,
            support_min=support_min,
            agg_mode=agg_mode,
            thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
            top_ws=top_ws, top_prec=top_prec, top_cp=top_cp,
        )

    # 3) Monta contexto para o LLM
    ctx_lines = [
        datasets_ctx,
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])  # evita None

    messages = [
        {
            "role": "system",
            "content": (
                "Você é o SAFETY • CHAT. Responda como especialista em Segurança Operacional (ESO), "
                "usando exclusivamente o contexto fornecido e padrões da organização."
            ),
        },
        {"role": "user", "content": user_input or "(sem texto)"},
        {"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
    ]

    # 4) Chamada ao modelo (Ollama Cloud)
    try:
        res = chat(messages, stream=False)
        answer = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        msg = str(e)
        if "405" in msg and "ollama.com/api" in msg:
            answer = (
                "Falha ao consultar o modelo (HTTP 405). "
                "Certifique-se de estar usando **POST https://ollama.com/api/chat** com cabeçalho "
                "`Authorization: Bearer <sua-chave>` e body JSON `{ \"model\": \"gpt-oss:20b-cloud\", \"messages\": [...] }`."
            )
        else:
            answer = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    ss["chat"].append({"role": "assistant", "content": answer})

# --------------------- Histórico --------------------------------------------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss["chat"][-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
