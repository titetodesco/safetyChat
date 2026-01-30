from __future__ import annotations
from typing import List, Tuple, Dict, Any
import pandas as pd


def _val(row: Any, key: str, default: str = "N/D") -> str:
    """Acessa `row[key]` com segurança para dict/Series; caso contrário retorna default."""
    if row is None:
        return default
    # pandas Series tem .get; dict também
    try:
        if hasattr(row, "get"):
            v = row.get(key, default)
        else:
            v = default
    except Exception:
        v = default
    s = "" if v is None else str(v)
    return s.strip() if s.strip() else default


def _first_present(row: Any, candidates: list[str], default: str = "N/D") -> str:
    """Retorna o primeiro campo existente/não-vazio em `row` dentre `candidates`."""
    for c in candidates:
        val = _val(row, c, default="")
        if val:
            return val
    return default


def hits_dataframe(hits: List[Tuple[str, float, Any]], loc_col: str | None) -> pd.DataFrame:
    """
    Constrói DataFrame exibido na UI a partir de hits em Sphera.
    Cada item de `hits` é (event_id, score, row).
    """
    rows = []
    for evid, score, row in hits or []:
        loc = _val(row, loc_col, "N/D") if loc_col else "N/D"
        desc = _first_present(row, ["Description", "DESCRIÇÃO", "DESCRIPTION"], default="")
        rows.append(
            {
                "EventID": evid,
                "Similaridade": round(float(score), 3),
                "LOCATION": loc,
                "Description": desc,
            }
        )
    return pd.DataFrame(rows)


def build_dic_matches_md(dic_res: Dict[str, list]) -> str:
    """
    Formata o bloco de dicionários agregados (WS/Prec/CP) para o contexto do LLM.
    Espera `dic_res` como {categoria: [(label, score), ...], ...}.
    """
    lines = ["=== DIC_MATCHES ==="]
    for k, arr in (dic_res or {}).items():
        if not arr:
            continue
        lines.append(f"## {k}")
        for lab, score in arr:
            try:
                lines.append(f"- {lab} (sim={float(score):.3f})")
            except Exception:
                lines.append(f"- {lab} (sim=NA)")
    return "\n".join(lines) + "\n"


def build_sphera_context_md(hits: List[Tuple[str, float, Any]], loc_col: str | None) -> str:
    """
    Formata o bloco Sphera para o contexto do LLM, com cabeçalho tabulado:
    EventID, Similaridade, LOCATION, Descrição (linha única).
    """
    lines = ["=== Sphera ===", "EventID\tSimilaridade\tLOCATION\tDescrição"]
    for evid, sim, row in hits or []:
        loc = _val(row, loc_col, "N/D") if loc_col else "N/D"
        desc = _first_present(row, ["Description", "DESCRIÇÃO", "DESCRIPTION"], default="").replace("\n", " ").strip()
        lines.append(f"{evid}\t{float(sim):.3f}\t{loc}\t{desc}")
    return "\n".join(lines) + "\n"


def build_gosee_context_md(hits: List[Tuple[str, float, Any]]) -> str:
    """
    Contexto para GoSee, com identificação, similaridade, local e observação.
    """
    if not hits:
        return ""
    lines = ["## GoSee — Observações semelhantes (contexto)\n"]
    for evid, sim, row in hits:
        rid = _first_present(row, ["ID", "Id", "id"], default=evid)
        loc = _first_present(row, ["Area", "AREA", "Location", "LOCATION"], default="N/D")
        obs = _first_present(row, ["Observation", "OBSERVATION", "Descrição", "DESCRIPTION"], default="N/D")
        lines.append(f"[GoSee/{rid}]  \nSimilaridade={float(sim):.3f}  |  Local={loc}\n")
        lines.append(f"Observation: {obs}\n")
    return "\n".join(lines)


def build_investigation_context_md(hits: List[Tuple[str, float, Any]]) -> str:
    """
    Contexto para relatórios de investigação (chunks).
    """
    if not hits:
        return ""
    lines = ["## Relatórios de Investigação — Trechos semelhantes (contexto)\n"]
    for evid, sim, row in hits:
        rid = _first_present(row, ["ID", "ChunkID", "DocID", "id"], default=evid)
        src = _first_present(row, ["SOURCE", "Source", "Documento", "Doc", "Arquivo"], default="Investigation")
        loc = _first_present(row, ["LOCATION", "Location", "Local", "Area", "AREA"], default="N/D")
        text = _first_present(
            row,
            ["TEXT", "text", "Content", "CONTENT", "DESCRIPTION", "Description", "EXCERPT", "Excerpt"],
            default="N/D",
        )
        lines.append(f"[Docs/{src}/{rid}]  \nSimilaridade={float(sim):.3f}  |  Local={loc}\n")
        lines.append(text + "\n")
    return "\n".join(lines)
