
from __future__ import annotations
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def ensure_st_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n

def encode_texts(encoder, texts, batch_size=64):
    import numpy as np
    if not texts:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype=np.float32)
    vecs = encoder.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
    return _l2norm(np.asarray(vecs, dtype=np.float32))

def encode_query(encoder, text: str):
    import numpy as np
    v = encoder.encode([text], show_progress_bar=False)[0].astype(np.float32)
    n = (np.linalg.norm(v) + 1e-9)
    return (v / n).astype(np.float32)
