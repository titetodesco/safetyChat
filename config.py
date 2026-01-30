from __future__ import annotations
import os
from pathlib import Path

# ---------------- Base folders ----------------
# Raiz do repositório (onde está este config.py)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AN_DIR   = DATA_DIR / "analytics"
XLSX_DIR = DATA_DIR / "xlsx"
DOCS_DIR = DATA_DIR / "docs"

# ---------------- Sphera ----------------
SPH_PQ_PATH  = AN_DIR / "sphera.parquet"
SPH_NPZ_PATH = AN_DIR / "sphera_embeddings.npz"
# URL pública do NPZ (ajuste se mudar o branch)
SPH_NPZ_URL  = "https://raw.githubusercontent.com/titetodesco/safetychat/main/data/analytics/sphera_embeddings.npz"

# ---------------- GoSee ----------------
GOSEE_PQ_PATH  = AN_DIR / "gosee.parquet"
GOSEE_NPZ_PATH = AN_DIR / "gosee_embeddings.npz"

# ---------------- Investigations / History ----------------
INC_JSONL_PATH = AN_DIR / "history_texts.jsonl"
INC_NPZ_PATH   = AN_DIR / "history_embeddings.npz"
INC_PQ_PATH    = None  # não há parquet para history

# ---------------- CP (taxonomia) ----------------
CP_NPZ_MAIN   = AN_DIR / "cp_embeddings.npz"
CP_NPZ_ALT    = None  # defina um caminho alternativo se existir
CP_LBL_PARQ   = AN_DIR / "cp_labels.parquet"
CP_LBL_JSONL  = AN_DIR / "cp_labels.jsonl"  # fallback

# ---------------- Dicionários (PT por padrão) ----------------
# Se você adicionar NPZ para WS/Precursores, ajuste aqui; enquanto não houver, deixe None.
WS_NPZ         = None  # AN_DIR / "ws_embeddings_pt.npz"
WS_LBL_PARQ    = AN_DIR / "ws_embeddings_pt.parquet"
WS_LBL_JSONL   = AN_DIR / "ws_embeddings_pt.jsonl"

PREC_NPZ       = None  # AN_DIR / "prec_embeddings_pt.npz"
PREC_LBL_PARQ  = AN_DIR / "prec_embeddings_pt.parquet"
PREC_LBL_JSONL = AN_DIR / "prec_embeddings_pt.jsonl"

# ---------------- Docs/Prompts ----------------
PROMPTS_MD_PATH        = DOCS_DIR / "prompts" / "prompts.md"
DATASETS_CONTEXT_PATH  = DATA_DIR / "datasets_context.md"

# ---------------- Modelos / Endpoints ----------------
# Ollama Cloud (pode sobrepor via secrets/env)
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "https://ollama.com/api")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Embeddings locais (usados só quando necessário; suas bases já têm embeddings pré-computados)
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

HEADERS_JSON = {"Content-Type": "application/json"}
if OLLAMA_API_KEY:
    HEADERS_JSON["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
