from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional

# --------------------------- Helpers leves ------------------------------------

def _unique_locs(df: Optional[pd.DataFrame], col: Optional[str]) -> List[str]:
    if df is None or col is None or col not in df.columns:
        return []
    return (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )

# ---------------------------- API pública -------------------------------------

def render_prompts_selector(
    prompts_bank: str | None, key_prefix: str = "sb_"
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Renderiza o seletor e o botão 'Carregar no rascunho'.
    As listas 'prompt_text_opts' e 'prompt_upl_opts' vêm prontas no session_state
    (carregadas pelo app a partir de prompts.md).
    """
    with st.sidebar.expander("Assistente de Prompts", expanded=False):
        txt_opts: List[str] = st.session_state.get("prompt_text_opts", [])
        upl_opts: List[str] = st.session_state.get("prompt_upl_opts", [])

        sel_text = st.selectbox(
            "Texto",
            options=txt_opts,
            index=None,
            placeholder="Selecione um prompt (Texto)",
            key=f"{key_prefix}sel_text_prompt",
        )
        sel_upl = st.selectbox(
            "Upload",
            options=upl_opts,
            index=None,
            placeholder="Selecione um prompt (Upload)",
            key=f"{key_prefix}sel_upl_prompt",
        )

        load_to_draft = st.button(
            "Carregar no rascunho",
            use_container_width=True,
            key=f"{key_prefix}btn_load_prompt",
        )

    return sel_text, sel_upl, load_to_draft


def render_retrieval_controls() -> Tuple[int, float, int]:
    st.sidebar.subheader("Recuperação – Sphera")
    k_sph = st.sidebar.slider(
        "Top-K Sphera", 5, 100, 20, step=5, key="sb_topk_sph"
    )
    thr_sph = st.sidebar.slider(
        "Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01, key="sb_thr_sph"
    )
    years = st.sidebar.slider(
        "Últimos N anos", 0, 10, 3, 1, key="sb_years"
    )
    return k_sph, thr_sph, years


def render_advanced_filters(
    df_sph: Optional[pd.DataFrame],
) -> Tuple[List[str], str, Optional[str], List[str]]:
    st.sidebar.subheader("Filtros avançados – Sphera")

    # A coluna de Location é definida no app e injetada aqui:
    loc_col: Optional[str] = st.session_state.get("sphera_loc_col", None)
    loc_opts = _unique_locs(df_sph, loc_col)

    substr = st.sidebar.text_input(
        "Description contém (substring)",
        value="",
        key="sb_desc_contains",
        help="Filtro case-insensitive; busca por substring em Description.",
    )

    locations = st.sidebar.multiselect(
        "Location (coluna: Location)" if loc_col else "Location (não detectada)",
        options=sorted(loc_opts) if loc_opts else [],
        default=[],
        key="sb_locations",
        placeholder="Escolha uma ou mais opções",
    )

    return locations, substr, loc_col, loc_opts


def render_aggregation_controls() -> Tuple[str, float, int, float, float, float, int, int, int]:
    st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")

    agg_mode = st.sidebar.selectbox(
        "Agregação",
        options=["max", "mean"],
        index=0,
        key="sb_agg_mode",
        help="Como consolidar a similaridade dos termos por evento (máx. ou média).",
    )

    per_event_thr = st.sidebar.slider(
        "Limiar por evento (dicionários)", 0.0, 1.0, 0.20, 0.01, key="sb_thr_event"
    )
    support_min = st.sidebar.slider(
        "Suporte mínimo (nº de eventos)", 1, 50, 3, 1, key="sb_support_min"
    )

    st.sidebar.markdown("**Limiares globais (após agregação)**")
    thr_ws = st.sidebar.slider(
        "Limiar global WS", 0.0, 1.0, 0.30, 0.01, key="sb_thr_ws"
    )
    thr_prec = st.sidebar.slider(
        "Limiar global Precursores", 0.0, 1.0, 0.30, 0.01, key="sb_thr_prec"
    )
    thr_cp = st.sidebar.slider(
        "Limiar global CP", 0.0, 1.0, 0.30, 0.01, key="sb_thr_cp"
    )

    st.sidebar.markdown("**Top-N por categoria**")
    top_ws = st.sidebar.slider("Top-N WS", 1, 50, 10, 1, key="sb_top_ws")
    top_prec = st.sidebar.slider("Top-N Precursores", 1, 50, 10, 1, key="sb_top_prec")
    top_cp = st.sidebar.slider("Top-N CP", 1, 50, 10, 1, key="sb_top_cp")

    return agg_mode, per_event_thr, support_min, thr_ws, thr_prec, thr_cp, top_ws, top_prec, top_cp


def render_util_buttons() -> Tuple[bool, bool]:
    st.sidebar.subheader("Utilitários")
    clear_upl = st.sidebar.button(
        "Limpar uploads", key="sb_clear_upl", use_container_width=True
    )
    clear_chat_btn = st.sidebar.button(
        "Limpar chat", key="sb_clear_chat", use_container_width=True
    )
    return clear_upl, clear_chat_btn
