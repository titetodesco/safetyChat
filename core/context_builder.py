from __future__ import annotations
from typing import List, Tuple, Any, Dict
import pandas as pd


def _to_str(x: Any, default: str = "N/D") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        # aceita "0,82" e "0.82"
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def _row_get(row: Any, key: str, default: str = "N/D") -> str:
    """Lê um campo de row (dict/Series/outros) com tolerância."""
    if row is None:
        return default
    try:
        if hasattr(row, "get"):
            return _to_str(row.get(key, default), default)
        # pandas Series: tem .get; outros objetos: tenta acesso por índice/atributo
        if isinstance(row, (list, tuple)) and key.isdigit():
            i = int(key)
            return _to_str(row[i], default) if 0 <= i < len(row) else default
    except Exception:
        pass
    return default


def _first_present(row: Any, keys: list[str], default: str = "N/D") -> str:
    for k in keys:
        v = _row_get(row, k, "")
        if v:
            return v
    return default


def hits_dataframe(hits: List[Tuple[str, float, Any]], loc_col: str | None) -> pd.DataFrame:
    """Monta DataFrame dos hits de Sphera, tolerante ao formato de cada linha."""
    rows = []
    for evid, s, row in (hits or []):
        sim = _to_float(s)
        loc = _row_get(row, loc_col, "N/D") if loc_col else "N/D"
        desc = _first_present(row, ["Description", "DESCRIÇÃO", "DESCRIPTION"], default="")
        rows.append({
            "EventID": evid,
            "Similaridade": round(sim, 3),
            "LOCATION": loc,
            "Description": desc,
        })
    return pd.DataFrame(rows)


def build_dic_matches_md(dic_res: Dict[str, list]) -> str:
    lines = ["=== DIC_MATCHES ==="]
    for k, arr in (dic_res or {}).items():
        if not arr:
            continue
        lines.append(f"## {k}")
        for lab, score in arr:
            lines.append(f"- {_to_str(lab)} (sim={_to_float(score):.3f})")
    return "\n".join(lines) + "\n"


def build_sphera_context_md(hits: List[Tuple[str, float, Any]], loc_col: str | None) -> str:
    lines = ["=== Sphera ===", "EventID\tSimilaridade\tLOCATION\tDescrição"]
    for evid, s, row in (hits or []):
        sim = _to_float(s)
        loc = _row_get(row, loc_col, "N/D") if loc_col else "N/D"
        desc = _first_present(row, ["Description", "DESCRIÇÃO", "DESCRIPTION"], default="").replace("\n", " ").strip()
        lines.append(f"{evid}\t{sim:.3f}\t{loc}\t{desc}")
    return "\n".join(lines) + "\n"


def build_gosee_context_md(hits: List[Tuple[str, float, Any]]) -> str:
    if not hits:
        return ""
    lines = ["## GoSee — Observações semelhantes (contexto)\n"]
    for evid, s, row in hits:
        sim = _to_float(s)
        rid = _first_present(row, ["ID", "Id", "id"], default=evid)
        loc = _first_present(row, ["Area", "AREA", "Location", "LOCATION"], default="N/D")
        obs = _first_present(row, ["Observation", "OBSERVATION", "Descrição", "DESCRIPTION"], default="N/D")
        lines.append(f"[GoSee/{rid}]  \nSimilaridade={sim:.3f}  |  Local={loc}\n")
        lines.append(f"Observation: {obs}\n")
    return "\n".join(lines)


def build_investigation_context_md(hits: List[Tuple[str, float, Any]]) -> str:
    if not hits:
        return ""
    lines = ["## Relatórios de Investigação — Trechos semelhantes (contexto)\n"]
    for evid, s, row in hits:
        sim = _to_float(s)
        rid = _first_present(row, ["ID", "ChunkID", "DocID", "id"], default=evid)
        src = _first_present(row, ["SOURCE", "Source", "Documento", "Doc", "Arquivo"], default="Investigation")
        loc = _first_present(row, ["LOCATION", "Location", "Local", "Area", "AREA"], default="N/D")
        text = _first_present(
            row, ["TEXT", "text", "Content", "CONTENT", "DESCRIPTION", "Description", "EXCERPT", "Excerpt"], default="N/D"
        )
        lines.append(f"[Docs/{src}/{rid}]  \nSimilaridade={sim:.3f}  |  Local={loc}\n")
        lines.append(text + "\n")
    return "\n".join(lines)
