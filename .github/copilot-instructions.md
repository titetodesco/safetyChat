# SafetyChat AI Coding Agent Instructions

## Project Architecture

**SafetyChat** is a Streamlit-based RAG (Retrieval-Augmented Generation) chat system for safety incident analysis. The application combines document upload/extraction, semantic search via embeddings, and LLM-powered chat responses using the Ollama Cloud API.

### Core Data Flow

1. **Data Loading** (`core/data_loader.py`): Loads pre-computed embedding indexes (`.npz` files) and metadata labels (`.jsonl`/`.parquet`) from `data/analytics/`
   - Multiple datasets: Sphera (incidents), GoSee, History (investigations), Precursors, and dictionaries (WS, CP)
   - All embeddings are pre-computed; no re-encoding of static data

2. **Semantic Search** (`core/sphera.py`): 
   - Converts user queries to embeddings using Sentence Transformers (`all-MiniLM-L6-v2` by default)
   - Performs cosine similarity search against loaded `.npz` embedding matrices
   - Returns top-K most similar events with scores and metadata rows

3. **Context Building** (`core/context_builder.py`):
   - Formats search results (hits) into markdown context
   - Extracts event IDs, locations, descriptions from hits dataframe
   - Aggregates dictionary matches over results for enrichment

4. **LLM Chat** (`services/llm_client.py`):
   - Calls Ollama Cloud API (`/api/chat` endpoint) with user prompt + retrieved context
   - Normalizes host URLs flexibly (handles various Ollama endpoint formats)
   - Requires `OLLAMA_API_KEY` in environment/secrets

5. **UI Layer** (`ui/main.py`, `app.py`):
   - Streamlit interface with textarea inputs, file upload, and chat display
   - Session state manages conversation history and draft prompts
   - File upload extracts text from PDF/DOCX/XLSX/CSV/TXT/MD (see `services/upload_extract.py`)

### Key Configuration

- **Environment variables** (see `config.py`):
  - `OLLAMA_HOST`: Ollama API endpoint (default: `https://ollama.com/api`)
  - `OLLAMA_MODEL`: Chat model name (default: `gpt-oss:20b-cloud`)
  - `OLLAMA_API_KEY`: Required for authentication
  - `DICT_LANG`: Language preference for dictionary matching (`"pt"` or `"en"`)

- **Data paths** anchored to `data/analytics/`:
  - Sphera: `sphera_embeddings.npz` (required), optional `.parquet` metadata
  - All other datasets follow pattern: `{name}_embeddings.npz` + `{name}_labels.jsonl`

## Development Workflows

### Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Set environment variables before running (especially `OLLAMA_API_KEY`, `OLLAMA_HOST`).

### File Upload & Extraction
- `services/upload_extract.py` provides `extract_any()` that handles all supported formats
- Extraction happens in-memory; no temporary files created
- Returned text is appended to `st.session_state.upld_texts` for chat context

### Adding New Datasets
1. Add new path constants in `config.py` (NPZ + labels pattern)
2. Create loader function in `core/data_loader.py` following the existing `load_sphera()` pattern
3. Reference in `app.py` and update UI inputs if needed

## Critical Patterns

### Error Handling & Resilience
- **Lenient field extraction** (`core/context_builder.py`): `_row_get()` handles missing columns gracefully with defaults ("N/D")
- **Event ID fallback** (`core/sphera.py`): Auto-detects EventID from multiple column name variants; generates synthetic IDs if needed
- **File type detection**: Case-insensitive; ignores unknown formats silently

### Column Name Flexibility
`app.py._ensure_eventid_column()` automatically normalizes EventID variants before processing. This pattern applies across data loading—column names are fuzzy-matched, not rigid.

### Embedding Model Configuration
- Model name retrieved via fallback chain in `core/sphera.py`: `OLLAMA_EMBEDDING_MODEL` → `EMBEDDING_MODEL` → hardcoded default
- Query encoding uses `core/encoding.ensure_st_encoder()` (Sentence Transformers cached encoder)
- Pre-computed embeddings assumed to be L2-normalized for cosine similarity

### Streamlit State Management
- `st.session_state` manages conversation history, uploaded texts, and draft prompts
- Clear/rerun pattern: modify state, call `st.rerun()` to re-render
- Session state keys: `"chat"`, `"upld_texts"`, `"draft_prompt"`

## Common Tasks

### Updating Search Parameters
- Top-K matches: controlled in `core/sphera.topk_similar()` call
- Similarity threshold: filter results after similarity computation
- Chunk size/overlap: if adding chunking logic, place in `core/context_builder.py`

### Adding LLM Parameters
- Override chat parameters via `**kwargs` in `services.llm_client.chat()` (e.g., `temperature`, `top_p`)
- Example: `chat(messages, temperature=0.7, stream=False)`

### Debugging Embeddings
- Check `.npz` structure: `np.load(..., allow_pickle=True)` to inspect keys
- Verify L2 normalization before similarity: `np.linalg.norm(vec) ≈ 1.0`
- Sentence Transformers default pool: `mean_pooling` of token embeddings

## Dependencies & Deployment

- **Python**: 3.10–3.11 (via `pyproject.toml`)
- **Heavy packages**: torch, transformers, sentence-transformers (used for local query encoding)
- **Streamlit Cloud**: Supports the full stack; secrets managed via dashboard
- **Git LFS**: Optionally track `.npz`, `.pdf`, `.docx` (see README)

## References
- Main app entry: [../app.py](../app.py)
- Core RAG logic: [../core/](../core/)
- UI structure: [../ui/main.py](../ui/main.py)
- Config & paths: [../config.py](../config.py)
