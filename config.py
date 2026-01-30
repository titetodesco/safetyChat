from __future__ import annotations
import os
from pathlib import Path

# ---------------- Pastas base ----------------
# Este arquivo está na RAIZ do repositório.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AN_DIR   = DATA_DIR / "analytics"
XLSX_DIR = DATA_DIR / "xlsx"
DOCS_DIR = DATA_DIR / "docs"

# ---------------- Sphera ----------------
SPH_PQ_PATH  = AN_DIR / "sphera.parquet"
SPH_NPZ_PATH = AN_DIR / "sphera_embeddings.npz"  # obrigatório

# ---------------- GoSee ----------------
GOSEE_PQ_PATH  = AN_DIR / "gosee.parquet"
GOSEE_NPZ_PATH = AN_DIR / "gosee_embeddings.npz"  # obrigatório para usar RAG em GoSee

# ---------------- Investigations / History ----------------
INC_JSONL_PATH = AN_DIR / "history_texts.jsonl"
INC_NPZ_PATH   = AN_DIR / "history_embeddings.npz"  # obrigatório p/ RAG de históricos
INC_PQ_PATH    = None  # se não tiver parquet do histórico, deixe None

# ---------------- Dicionários (NPZ obrigatórios) ----------------
# Ajuste os nomes conforme os arquivos que você versionou no repo.
WS_NPZ         = AN_DIR / "ws_embeddings_pt.npz"
WS_LBL_PARQ    = AN_DIR / "ws_embeddings_pt.parquet"
WS_LBL_JSONL   = AN_DIR / "ws_embeddings_pt.jsonl"

PREC_NPZ       = AN_DIR / "prec_embeddings_pt.npz"
PREC_LBL_PARQ  = AN_DIR / "prec_embeddings_pt.parquet"
PREC_LBL_JSONL = AN_DIR / "prec_embeddings_pt.jsonl"

CP_NPZ_MAIN    = AN_DIR / "cp_embeddings.npz"
CP_NPZ_ALT     = None  # defina um alternativo se existir
CP_LBL_PARQ    = AN_DIR / "cp_labels.parquet"
CP_LBL_JSONL   = AN_DIR / "cp_labels.jsonl"

# ---------------- Docs/Prompts ----------------
PROMPTS_MD_PATH        = DOCS_DIR / "prompts" / "prompts.md"
DATASETS_CONTEXT_PATH  = DATA_DIR / "datasets_context.md"

# ---------------- Modelo (Ollama Cloud) ----------------
# Use secrets ou variáveis de ambiente em produção.
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "https://ollama.com/api")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

HEADERS_JSON = {"Content-Type": "application/json"}
if OLLAMA_API_KEY:
    HEADERS_JSON["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
