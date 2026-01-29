from __future__ import annotations
import os
from pathlib import Path

# Base folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AN_DIR   = DATA_DIR / "analytics"
XLSX_DIR = DATA_DIR / "xlsx"
DOCS_DIR = DATA_DIR / "docs"

# Files (keep names compatible with your repo)
SPH_PQ_PATH = AN_DIR / "sphera.parquet"
SPH_NPZ_PATH = AN_DIR / "sphera_embeddings.npz"

# GoSee
GOSEE_PQ_PATH  = AN_DIR / "gosee.parquet"            # <- ajuste se seu nome for outro
GOSEE_NPZ_PATH = AN_DIR / "gosee_embeddings.npz"     # <- idem

# Relatórios de investigação
INC_PQ_PATH  = AN_DIR / "investigations.parquet"     # <- ajuste se seu nome for outro
INC_NPZ_PATH = AN_DIR / "investigations_embeddings.npz"

PROMPTS_MD_PATH = DATA_DIR / "prompts" / "prompts.md"
DATASETS_CONTEXT_PATH = DOCS_DIR / "datasets_context.md"

# Dictionaries (PT by default)
WS_NPZ   = AN_DIR / "ws_embeddings_pt.npz"
WS_LBL   = AN_DIR / "ws_embeddings_pt.parquet"
PREC_NPZ = AN_DIR / "prec_embeddings_pt.npz"
PREC_LBL = AN_DIR / "prec_embeddings_pt.parquet"
CP_NPZ_MAIN  = AN_DIR / "cp_embeddings.npz"
CP_NPZ_ALT   = AN_DIR / "cp_vectors.npz"           # fallback
CP_LBL_PARQ  = AN_DIR / "cp_labels.parquet"
CP_LBL_JSONL = AN_DIR / "cp_labels.jsonl"          # fallback

# Models / endpoints
# Ollama Cloud defaults (you can override via secrets or env)
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "https://ollama.com/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Embedding via SentenceTransformers (local, precomputed embeddings are used for Sphera/Dicts)
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

HEADERS_JSON = {"Content-Type": "application/json"}
if OLLAMA_API_KEY:
    HEADERS_JSON["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
