# core/encoding.py
from __future__ import annotations
from typing import Iterable, Optional, List
import numpy as np

# Cache simples de encoder (evita recarregar o modelo)
_ENCODER = None
_ENCODER_NAME: Optional[str] = None

@st.cache_resource(show_spinner=False)
def ensure_st_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Carrega e cacheia um SentenceTransformer. Sempre retorna um objeto com `.encode`."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return model

def encode_texts(encoder, texts):
    """
    Retorna matriz [n x d] em float32, normalizada por linha.
    Funciona com SentenceTransformer (tem .encode) ou com encoder 'callable'.
    """
    if hasattr(encoder, "encode"):
        # Sentence-Transformers
        vecs = encoder.encode(
            texts,
            convert_to_numpy=True,     # retorna np.ndarray
            normalize_embeddings=False # normalizamos manualmente por consistência
        )
        V = np.asarray(vecs, dtype=np.float32)
    else:
        # Encoder 'callable' que retorna vetores para cada texto
        rows = []
        for t in texts:
            v = np.asarray(encoder(t), dtype=np.float32)
            rows.append(v)
        V = np.vstack(rows).astype(np.float32)

    # normalização L2 por linha
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V = V / norms
    return V

def encode_query(text, encoder):
    """
    Retorna vetor [d] float32 normalizado.
    Compatível com SentenceTransformer ou encoder 'callable'.
    """
    if hasattr(encoder, "encode"):
        v = encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False
        )[0]
        v = np.asarray(v, dtype=np.float32)
    else:
        v = np.asarray(encoder(text), dtype=np.float32)

    v /= (np.linalg.norm(v) + 1e-12)
    return v
