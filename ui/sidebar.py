# ui/sidebar.py
from __future__ import annotations
import streamlit as st
from typing import List, Tuple
import pandas as pd
from core.sphera import get_sphera_location_col, location_options

def render_prompts_selector(prompts_bank: str):
    st.sidebar.subheader("Assistente de Prompts")
    col1, col2 = st.sidebar.columns(2)

    txt_opts = [
        "(1) Síntese focada em lições",
        "(2) Recomendações",
        "(3) Padrões recorrentes",
        "(4) Análise causal",
        "(5) Resumo executivo",
    ]
    upl_opts = [
        "(1) Destacar achados do upload",
        "(2) Comparar com Sphera",
        "(3) Lacunas e riscos",
        "(4) Checklists",
        "(5) Perguntas ao time",
    ]

    # chaves de estado para lembrar a escolha do usuário
    st.session_state.setdefault("sel_text_prompt", None)
    st.session_state.setdefault("sel_upl_prompt", None)

    # Streamlit >= 1.25 permite index=None + placeholder
    sel_text = col1.selectbox(
        "Texto",
        options=txt_opts,
        index=None,
        placeholder="Selecione um prompt…",
        key="sel_text_prompt",
    )
    sel_upl = col2.selectbox(
        "Upload",
        options=upl_opts,
        index=None,
        placeholder="Selecione um prompt…",
        key="sel_upl_prompt",
    )

    # Botão só fica ativo se pelo menos um for escolhido
    disable_btn = (sel_text is None) and (sel_upl is None)
    load_btn = st.sidebar.button("Carregar no rascunho", use_container_width=True, disabled=disable_btn)

    return (sel_text, sel_upl, load_btn)
