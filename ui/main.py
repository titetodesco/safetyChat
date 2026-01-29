# ui/main.py
from __future__ import annotations

import streamlit as st
from typing import Tuple, List

# Carregadores centrais (já existentes no projeto)
from config import DATASETS_CONTEXT_PATH, PROMPTS_MD_PATH
from core.data_loader import (
    load_sphera,          # retorna (df_sph, E_sph)
    load_prompts_md,      # retorna str
    load_datasets_context # retorna str
)
from services.upload_extract import extract_any


def _init_base_state() -> None:
    """Garante chaves essenciais do estado antes de qualquer widget."""
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("upld_texts", [])
    st.session_state.setdefault("draft_prompt", "")


def _render_main_inputs() -> Tuple[bool, str | None, str | None]:
    """
    Renderiza a ÁREA PRINCIPAL (sem sidebar):
      - textarea única: 'Conteúdo do prompt' (ligada a st.session_state['draft_prompt'])
      - textarea: 'Texto de análise (para Sphera)'
      - file_uploader: upload opcional
      - botões: Enviar / Limpar rascunho / Limpar chat

    Retorna:
      (go_btn, user_text, uploaded_name)
    """
    st.title("SAFETY • CHAT")

    # --- textarea ÚNICA controlada por estado ---
    st.text_area(
        "Conteúdo do prompt",
        key="draft_prompt",
        height=220,
        placeholder="Escreva aqui sua solicitação...",
    )

    user_text = st.text_area(
        "Texto de análise (para Sphera)",
        key="user_text",
        height=160,
        placeholder="Descreva o cenário...",
    )

    upl = st.file_uploader(
        "Anexar arquivo (opcional)",
        type=["pdf", "docx", "xlsx", "txt", "md", "csv"],
        accept_multiple_files=False,
    )
    uploaded_name = None
    if upl is not None:
        text = extract_any(upl)
        if text:
            st.session_state.upld_texts.append(text)
            uploaded_name = upl.name
            st.success(f"Upload recebido: {uploaded_name}")
        else:
            st.warning(f"Não foi possível extrair texto de {upl.name}.")

    col_run1, col_run2, col_run3 = st.columns([1, 1, 1])
    go_btn      = col_run1.button("Enviar para o chat", type="primary", use_container_width=True)
    clear_draft = col_run2.button("Limpar rascunho", use_container_width=True)
    clear_chat  = col_run3.button("Limpar chat", use_container_width=True)

    # Ações de limpeza: atualizam estado e rerenderizam a página
    if clear_draft:
        st.session_state["draft_prompt"] = ""
        st.rerun()
    if clear_chat:
        st.session_state["chat"] = []
        st.rerun()

    return go_btn, user_text, uploaded_name


def render_main() -> Tuple[
    bool,                 # go_btn
    str | None,           # user_text
    "pd.DataFrame | None",# df_sph
    "any | None",         # E_sph (np.ndarray ou similar)
    str,                  # datasets_ctx (md)
    str,                  # prompts_md (md)
    List[str],            # upl_texts
]:
    """
    Função de entrada usada pelo app.py.
    NÃO cria nada no sidebar. A sidebar deve ser renderizada por ui/sidebar.py.
    Retorna exatamente a tupla esperada pelo app.py.
    """
    _init_base_state()

    # Contextos fixos (sempre injetados pelo app): md de datasets e banco de prompts
    datasets_ctx = load_datasets_context(DATASETS_CONTEXT_PATH)
    prompts_md   = load_prompts_md(PROMPTS_MD_PATH)

    # UI principal (sem assistente de prompt aqui)
    go_btn, user_text, _ = _render_main_inputs()

    # Carrega Sphera (parquet + npz já existentes em data/analytics)
    df_sph, E_sph = load_sphera()

    # Retorno no formato esperado pelo app.py
    return (
        go_btn,
        user_text,
        df_sph,
        E_sph,
        datasets_ctx,
        prompts_md,
        st.session_state.upld_texts,
    )
