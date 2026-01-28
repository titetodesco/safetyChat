
from __future__ import annotations
import streamlit as st
import pandas as pd

def show_hits_table(df_hits: pd.DataFrame):
    st.dataframe(df_hits, use_container_width=True, hide_index=True)

def show_debug_raw(debug: dict):
    with st.expander("🔎 Depuração — Top-N brutos (ignora thresholds)", expanded=False):
        for k, arr in debug.items():
            if not arr: 
                continue
            st.markdown(f"**{k}**")
            st.dataframe(pd.DataFrame(arr, columns=["Termo","Score"]), use_container_width=True, hide_index=True)
