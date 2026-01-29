from __future__ import annotations
import streamlit as st
import pandas as pd

from config import DATASETS_CONTEXT_PATH, PROMPTS_MD_PATH
from core.data_loader import load_sphera, load_prompts_md, load_datasets_context, load_dicts
from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.dictionaries import aggregate_dict_matches_over_hits
from core.context_builder import hits_dataframe, build_dic_matches_md, build_sphera_context_md
from services.upload_extract import extract_any
from services.llm_client import chat

def render_main():
    st.title("SAFETY • CHAT ")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "upld_texts" not in st.session_state:
        st.session_state.upld_texts = []
    if "draft_prompt" not in st.session_state:
        st.session_state.draft_prompt = ""

    datasets_ctx = load_datasets_context(DATASETS_CONTEXT_PATH)
    prompts_md = load_prompts_md(PROMPTS_MD_PATH)

    # ui/main.py (dentro de render_main)
    if "draft_prompt" not in st.session_state:
        st.session_state["draft_prompt"] = ""

    st.text_area("Conteúdo do prompt", key="draft_prompt", height=220, placeholder="Escreva aqui sua solicitação...")
    user_text = st.text_area("Texto de análise (para Sphera)", key="user_text", height=160, placeholder="Descreva o cenário...")
    upl = st.file_uploader("Anexar arquivo (opcional)", type=["pdf","docx","xlsx","txt","md","csv"], accept_multiple_files=False)

    
    # renderiza o text_area usando outro key
    draft_val = st.session_state.get("draft_prompt", "")
    new_val = st.text_area(
        "Conteúdo do prompt",
        value=draft_val,
        key="draft_prompt_input",           # <- key diferente
        height=160,
        placeholder="Escreva aqui sua solicitação..."
    )
    
    # sincroniza de volta se o usuário editar manualmente
    if new_val != st.session_state["draft_prompt"]:
        st.session_state["draft_prompt"] = new_val

    user_text = st.text_area("Texto de análise (para Sphera)", key="user_text", height=120, placeholder="Descreva o cenário...")

    upl = st.file_uploader("Anexar arquivo (opcional)", type=["pdf","docx","xlsx","csv","txt","md"])
    if upl is not None:
        text = extract_any(upl)
        if text:
            st.session_state.upld_texts.append(text)
            st.success(f"Upload recebido: {upl.name}")
        else:
            st.warning(f"Não foi possível extrair texto de {upl.name}.")

    col_run1, col_run2, col_run3 = st.columns([1,1,1])
    go_btn      = col_run1.button("Enviar para o chat", type="primary", use_container_width=True)
    clear_draft = col_run2.button("Limpar rascunho", use_container_width=True)
    clear_chat  = col_run3.button("Limpar chat", use_container_width=True)

    if clear_draft:
        st.session_state.draft_prompt = ""
        st.stop()
    if clear_chat:
        st.session_state.chat = []
        st.stop()

    df_sph, E_sph = load_sphera()
    return go_btn, user_text, df_sph, E_sph, datasets_ctx, prompts_md, st.session_state.upld_texts
