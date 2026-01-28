
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
