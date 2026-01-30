# core/encoding.py
from __future__ import annotations
from typing import Iterable, Optional, List
import numpy as np

# Cache simples de encoder (evita recarregar o modelo)
_ENCODER = None
_ENCODER_NAME: Optional[str] = None

def ensure_st_encoder(model_name: str):
    """
    Garante um SentenceTransformer carregado com o nome informado.
    Reutiliza a instância se o nome não mudar.
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
    Codifica um único texto -> vetor numpy 1D.
    NÃO usa `show_progress_bar` para manter compatibilidade.
    """
    if not text:
        # Fallback só é usado se text vier vazio; dimensão real do modelo será
        # a do encode abaixo, então evitamos adivinhar aqui.
        return np.zeros((0,), dtype="float32")

    vec = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,  # normalizamos no consumidor quando preciso
        # show_progress_bar NÃO suportado em algumas versões
    )[0]
    return vec

def encode_texts(texts: Iterable[str], encoder, batch_size: int = 64) -> np.ndarray:
    """
    Codifica uma lista/iterável de textos -> matriz numpy 2D [N, D].
    Compatível com versões antigas do sentence-transformers (sem show_progress_bar).
    """
    # Normaliza entrada
    if texts is None:
        texts = []
    if not isinstance(texts, list):
        texts = list(texts)

    if len(texts) == 0:
        # Sem textos -> matriz vazia [0, D]; descobrimos D com um dummy curto
        # para evitar adivinhação da dimensão.
        dummy = encoder.encode([""], convert_to_numpy=True, normalize_embeddings=False)[0]
        return np.zeros((0, dummy.shape[0]), dtype=dummy.dtype)

    mat = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        batch_size=batch_size,
        # show_progress_bar NÃO suportado em algumas versões
    )
    return mat
