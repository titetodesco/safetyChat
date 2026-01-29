from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

from config import SPH_PQ_PATH, WS_LBL, PREC_LBL, CP_LBL_PARQ
from core.data_loader import load_parquet_safe

st.set_page_config(page_title="Data Viewer", layout="wide")

st.title("📊 Data Viewer")
st.markdown("Preview and explore incident and dictionary datasets.")


def download_dataframe(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
        use_container_width=True
    )


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    
    df_filtered = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) > 1 and len(unique_vals) <= 50:
                selected = st.multiselect(
                    f"Filter {col}",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"filter_{col}_{id(df)}"
                )
                if selected:
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]
        elif pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            if min_val != max_val:
                selected_range = st.slider(
                    f"Filter {col}",
                    min_val, max_val, (min_val, max_val),
                    key=f"filter_range_{col}_{id(df)}"
                )
                df_filtered = df_filtered[
                    (df_filtered[col] >= selected_range[0]) & 
                    (df_filtered[col] <= selected_range[1])
                ]
    
    return df_filtered


tabs = st.tabs(["Incidents", "Dictionaries"])


with tabs[0]:
    st.subheader("🚨 Sphera Incidents")
    
    df_sphera = load_parquet_safe(SPH_PQ_PATH)
    
    if df_sphera is not None and not df_sphera.empty:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.metric("Total Rows", len(df_sphera))
            download_dataframe(df_sphera, "sphera_incidents.csv", "📥 Download Full Dataset")
        
        with st.expander("🔍 Filters", expanded=False):
            df_sphera_filtered = filter_dataframe(df_sphera)
        
        st.dataframe(
            df_sphera_filtered,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        st.caption(f"Showing {len(df_sphera_filtered)} of {len(df_sphera)} rows")
    else:
        st.warning("Sphera dataset not found or empty.")
    
    st.markdown("---")
    
    st.subheader("👁️ GoSee Observations")
    gosee_path = SPH_PQ_PATH.parent / "gosee.parquet"
    df_gosee = load_parquet_safe(gosee_path)
    
    if df_gosee is not None and not df_gosee.empty:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.metric("Total Rows", len(df_gosee))
            download_dataframe(df_gosee, "gosee_observations.csv", "📥 Download Full Dataset")
        
        with st.expander("🔍 Filters", expanded=False):
            df_gosee_filtered = filter_dataframe(df_gosee)
        
        st.dataframe(
            df_gosee_filtered,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        st.caption(f"Showing {len(df_gosee_filtered)} of {len(df_gosee)} rows")
    else:
        st.warning("GoSee dataset not found or empty.")


with tabs[1]:
    st.subheader("⚠️ Weak Signals Dictionary")
    
    df_ws = load_parquet_safe(WS_LBL)
    
    if df_ws is not None and not df_ws.empty:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.metric("Total Rows", len(df_ws))
            download_dataframe(df_ws, "weak_signals_dictionary.csv", "📥 Download")
        
        with st.expander("🔍 Filters", expanded=False):
            df_ws_filtered = filter_dataframe(df_ws)
        
        st.dataframe(
            df_ws_filtered,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        st.caption(f"Showing {len(df_ws_filtered)} of {len(df_ws)} rows")
    else:
        st.warning("Weak Signals dictionary not found or empty.")
    
    st.markdown("---")
    
    st.subheader("🎯 Precursors Dictionary")
    
    df_prec = load_parquet_safe(PREC_LBL)
    
    if df_prec is not None and not df_prec.empty:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.metric("Total Rows", len(df_prec))
            download_dataframe(df_prec, "precursors_dictionary.csv", "📥 Download")
        
        with st.expander("🔍 Filters", expanded=False):
            df_prec_filtered = filter_dataframe(df_prec)
        
        st.dataframe(
            df_prec_filtered,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        st.caption(f"Showing {len(df_prec_filtered)} of {len(df_prec)} rows")
    else:
        st.warning("Precursors dictionary not found or empty.")
    
    st.markdown("---")
    
    st.subheader("📋 Control Points Taxonomy")
    
    df_cp = load_parquet_safe(CP_LBL_PARQ)
    
    if df_cp is not None and not df_cp.empty:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.metric("Total Rows", len(df_cp))
            download_dataframe(df_cp, "cp_taxonomy.csv", "📥 Download")
        
        with st.expander("🔍 Filters", expanded=False):
            df_cp_filtered = filter_dataframe(df_cp)
        
        st.dataframe(
            df_cp_filtered,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        st.caption(f"Showing {len(df_cp_filtered)} of {len(df_cp)} rows")
    else:
        st.warning("Control Points taxonomy not found or empty.")


st.sidebar.header("About")
st.sidebar.info(
    "This page provides interactive previews of the incident and dictionary datasets "
    "used by the Safety Chat application. Use the filters to explore the data and "
    "download buttons to export CSV files."
)
