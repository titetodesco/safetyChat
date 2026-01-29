from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd

def hits_dataframe(hits: List[Tuple[str,float,pd.Series]], loc_col: str|None) -> pd.DataFrame:
    rows = []
    for evid, s, row in hits:
        loc = row.get(loc_col, "N/D") if loc_col else "N/D"
        desc = str(row.get("Description",""))
        rows.append({"EventID": evid, "Similaridade": round(float(s),3), "LOCATION": str(loc), "Description": desc})
    return pd.DataFrame(rows)

def build_dic_matches_md(dic_res: Dict[str, list]) -> str:
    lines = ["=== DIC_MATCHES ==="]
    for k, arr in dic_res.items():
        if not arr:
            continue
        lines.append(f"## {k}")
        for lab, score in arr:
            lines.append(f"- {lab} (sim={score:.3f})")
    return "\n".join(lines) + "\n"

def build_sphera_context_md(hits: List[Tuple[str,float,pd.Series]], loc_col: str|None) -> str:
    lines = ["=== Sphera ===", "EventID\tSimilaridade\tLOCATION\tDescrição"]
    for evid, s, row in hits:
        loc = row.get(loc_col, "N/D") if loc_col else "N/D"
        desc = str(row.get("Description","")).replace("\n"," ").strip()
        lines.append(f"{evid}\t{s:.3f}\t{loc}\t{desc}")
    return "\n".join(lines) + "\n"
def _first_present(row: pd.Series, candidates: list[str], default: str = "N/D") -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return default

def build_gosee_context_md(hits: List[Tuple[str, float, pd.Series]]) -> str:
    """
    Formata blocos [GoSee/<ID>] com:
    - ID (ou índice se não houver coluna ID),
    - Similaridade (0–1),
    - Local (coluna Area/AREA/Location/LOCATION se existir),
    - Observação completa (Observation).
    """
    if not hits:
        return ""
    lines = []
    lines.append("## GoSee — Observações semelhantes (contexto)\n")
    for evid, sim, row in hits:
        rid = _first_present(row, ["ID", "Id", "id"], default=evid)
        loc = _first_present(row, ["Area", "AREA", "Location", "LOCATION"], default="N/D")
        obs = _first_present(row, ["Observation", "OBSERVATION", "Descrição", "DESCRIPTION"], default="N/D")
        lines.append(f"[GoSee/{rid}]  \nSimilaridade={sim:.3f}  |  Local={loc}\n")
        lines.append(f"Observation: {obs}\n")
    return "\n".join(lines)

def build_investigation_context_md(hits: List[Tuple[str, float, pd.Series]]) -> str:
    """
    Formata blocos [Docs/Investigation/<ID>] para relatórios de investigação:
    tenta usar campos textuais longos comuns em pipelines de chunking (TEXT, CONTENT, DESCRIPTION, EXCERPT).
    """
    if not hits:
        return ""
    lines = []
    lines.append("## Relatórios de Investigação — Trechos semelhantes (contexto)\n")
    for evid, sim, row in hits:
        rid = _first_present(row, ["ID", "ChunkID", "DocID", "id"], default=evid)
        src = _first_present(row, ["SOURCE", "Source", "Documento", "Doc", "Arquivo"], default="Investigation")
        loc = _first_present(row, ["LOCATION", "Location", "Local", "Area", "AREA"], default="N/D")
        text = _first_present(row, ["TEXT", "Content", "CONTENT", "DESCRIPTION", "Description", "EXCERPT", "Excerpt"], default="N/D")
        lines.append(f"[Docs/{src}/{rid}]  \nSimilaridade={sim:.3f}  |  Local={loc}\n")
        lines.append(text + "\n")
    return "\n".join(lines)
