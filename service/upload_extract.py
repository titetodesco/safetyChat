
from __future__ import annotations
import io, pandas as pd

def extract_pdf_text(file) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""

def extract_docx_text(file) -> str:
    try:
        from docx import Document
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_xlsx_text(file) -> str:
    try:
        dfs = pd.read_excel(file, sheet_name=None)
        parts = []
        for name, df in dfs.items():
            parts += df.astype(str).fillna("").apply(lambda r: " ".join(r.values), axis=1).tolist()
        return "\n".join(parts)
    except Exception:
        return ""

def extract_csv_text(file) -> str:
    try:
        df = pd.read_csv(file)
        return "\n".join(df.astype(str).fillna("").apply(lambda r: " ".join(r.values), axis=1).tolist())
    except Exception:
        return ""

def extract_any(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    bio = io.BytesIO(data)
    if name.endswith(".pdf"):
        return extract_pdf_text(bio)
    if name.endswith(".docx"):
        return extract_docx_text(bio)
    if name.endswith(".xlsx"):
        return extract_xlsx_text(bio)
    if name.endswith(".csv"):
        return extract_csv_text(bio)
    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return ""
