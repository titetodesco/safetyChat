# core/encoding.py
from __future__ import annotations
from typing import Optional
import numpy as np

# Carregador global com cache leve (evita rebaixar o modelo toda hora)
_ENCODER = None
_ENCODER_NAME: Optional[str] = None

def ensure_st_encoder(model_name: str):
    """
    Garante um SentenceTransformer carregado com o nome informado.
    Reusa o mesmo encoder se o nome não mudar.
    """
    global _ENCODER, _ENCODER_NAME
    if _ENCODER is not None and _ENCODER_NAME == model_name:
        return _ENCODER

    from sentence_transformers import SentenceTransformer
    _ENCODER = SentenceTransformer(model_name)
    _ENCODER_NAME = model_name
    return _ENCODER

def encode_query(text: str, encoder) -> np.ndarray:
    """
    Codifica um único texto -> vetor numpy (1D).
    **Sem** show_progress_bar (compatível com versões antigas).
    """
    if not text:
        return np.zeros((384,), dtype="float32")  # fallback seguro; ajuste o dim se necessário
    # Retorna um vetor 1D
    vec = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,  # normalizamos no consumidor (sphera.py)
        # NÃO passe show_progress_bar aqui
    )[0]
    return vec
