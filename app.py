from __future__ import annotations
import streamlit as st
import pandas as pd

# --- Config e serviços
from config import DATASETS_CONTEXT_PATH, PROMPTS_MD_PATH
from core.data_loader import (
    load_sphera, load_prompts_md, load_datasets_context, load_dicts,
)
from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.context_builder import hits_dataframe, build_dic_matches_md, build_sphera_context_md
from core.dictionaries import aggregate_dict_matches_over_hits
from services.upload_extract import extract_any
from services.llm_client import chat

# --- Sidebar (SEM assistente de prompts)
from ui.sidebar import (
    render_retrieval_controls,
    render_advanced_filters,
    render_aggregation_controls,
    render_util_buttons,
)

st.set_page_config(page_title="SAFETY • CHAT", layout="wide")


st.write("config carregado de:", config.__file__)
st.write("SPH_PQ_PATH:", str(config.SPH_PQ_PATH), isinstance(config.SPH_PQ_PATH, Path), config.SPH_PQ_PATH.exists())
st.write("SPH_NPZ_PATH:", str(config.SPH_NPZ_PATH), isinstance(config.SPH_NPZ_PATH, Path), config.SPH_NPZ_PATH.exists())



# Se você também monta um 'sphera_context_md' e 'dic_matches_md':
try:
    from core.context_builder import build_sphera_context_md, build_dic_matches_md
    if 'hits' in locals():
        sph_ctx = build_sphera_context_md(hits, get_sphera_location_col(df_sph))
        st.write("len(sphera_context_md) =", len(sph_ctx or ""))
        st.code((sph_ctx or "")[:500])
    # Se tiver dic_res:
    if 'dic_res' in locals():
        dic_ctx = build_dic_matches_md(dic_res)
        st.write("len(dic_matches_md) =", len(dic_ctx or ""))
        st.code((dic_ctx or "")[:500])
except Exception as e:
    st.warning(f"(Depuração contexto) Falha ao montar previews: {e}")     


# --------------------- Estado base (sempre antes dos widgets) ---------------------
ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("chat", [])
ss.setdefault("upld_texts", [])

# --------------------- Carregamentos (dados, contexto) ---------------------------
datasets_ctx = load_datasets_context(DATASETS_CONTEXT_PATH) or ""
# (prompts_md carregado mas não usado; mantemos para compatibilidade com funções futuras)
_ = load_prompts_md(PROMPTS_MD_PATH)

df_sph, E_sph = load_sphera()  # depende do config.py correto!

# --------------------- SIDEBAR (parâmetros) --------------------------------------
with st.sidebar:
    st.caption("Parâmetros RAG e utilitários")

k_sph, thr_sph, years = render_retrieval_controls()
locations, substr, loc_col_guess, loc_options = render_advanced_filters(df_sph)
(agg_mode, per_event_thr, support_min, thr_ws, thr_prec, thr_cp,
 top_ws, top_prec, top_cp) = render_aggregation_controls()
clear_upl, clear_chat_btn = render_util_buttons()

# Ações utilitárias
if clear_upl:
    ss["upld_texts"] = []
    st.success("Uploads limpos.")
if clear_chat_btn:
    ss["chat"] = []
    st.success("Chat limpo.")

# --------------------- UI principal (centro) --------------------------------------
st.title("SAFETY • CHAT")

draft = st.text_area("Conteúdo do prompt", key="draft_prompt", height=220)

txt_for_sph = st.text_area(
    "Texto de análise (para Sphera)",
    placeholder="Descreva o cenário...", height=180, key="analysis_text"
)

upl = st.file_uploader(
    "Anexar arquivo (opcional)",
    type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
    accept_multiple_files=True, key="uploader_any"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
go_btn       = col_btn1.button("Enviar para o chat", use_container_width=True, key="btn_go")
clear_draft  = col_btn2.button("Limpar rascunho", use_container_width=True, key="btn_clear_draft")
clear_chat   = col_btn3.button("Limpar chat", use_container_width=True, key="btn_clear_chat_msg")

if clear_draft:
    ss["draft_prompt"] = ""
    st.rerun()
if clear_chat:
    ss["chat"] = []
    st.rerun()

# Extrair textos de uploads (se houver)
if upl:
    extracted = []
    for f in upl:
        try:
            text = extract_any(f)
            if text:
                extracted.append(text)
        except Exception as e:
            st.warning(f"Falha ao extrair de {getattr(f,'name','(arquivo)')}: {e}")
    ss["upld_texts"] = extracted

# --------------------- Execução (ao clicar) ---------------------------------------
if go_btn:
    user_input = (ss.get("draft_prompt") or "").strip()

    # 1) Recuperação Sphera (RAG)
    loc_col_eff = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    hits = []
    if isinstance(df_base, pd.DataFrame) and E_sph is not None and user_input:
        # Importante: topk_similar usa o MESMO encoder dos embeddings pré-calculados
        hits = topk_similar(user_input, df_base, E_sph, topk=k_sph, min_sim=thr_sph)

    st.subheader(f"Eventos do Sphera (Top-{min(k_sph, len(hits))})")
    if hits:
        st.dataframe(hits_dataframe(hits, loc_col_eff), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum evento recuperado. Ajuste o texto/limiar/Top-K.")

    # 2) Agregação de dicionários (só se houver hits)
    dic_res, debug_raw = {}, {}
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()
        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
            per_event_thr=per_event_thr, support_min=support_min, agg_mode=agg_mode,
            thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
            top_ws=top_ws, top_prec=top_prec, top_cp=top_cp
        )

    # 3) Contexto para o LLM (dados + matches)
    ctx_lines = [
        datasets_ctx,
        build_sphera_context_md(hits, loc_col_eff),
        build_dic_matches_md(dic_res),
    ]
    ctx_full = "\n".join([c for c in ctx_lines if c])

    # 4) Chamada ao modelo
    messages = [
        {"role": "system", "content": "Você é o SAFETY • CHAT. Responda como especialista em ESO usando o contexto abaixo."},
        {"role": "user", "content": user_input},
        {"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + ctx_full},
    ]

    try:
        res = chat(messages, stream=False)
        answer = res.get("message", {}).get("content", "(sem conteúdo)")
    except Exception as e:
        msg = str(e)
        if "405" in msg and "ollama.com/api" in msg:
            answer = (
                "Falha ao consultar o modelo (HTTP 405). Verifique em services/llm_client.py "
                "se está usando **POST https://ollama.com/api/chat** com Bearer API-Key e body JSON {model, messages}."
            )
        else:
            answer = f"Falha ao consultar o modelo: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    ss["chat"].append({"role": "assistant", "content": answer})

# --------------------- Histórico -----------------------------------------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss["chat"][-10:]:
        with st.chat_message("assistant"):
            st.markdown(m.get("content", ""))
