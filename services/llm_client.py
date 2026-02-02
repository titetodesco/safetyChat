from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

from config import OLLAMA_HOST, OLLAMA_MODEL, HEADERS_JSON


def _normalize_chat_url(host: str) -> str:
    """
    Aceita host como:
      - https://ollama.com
      - https://ollama.com/api
      - http://localhost:11434
      - http://localhost:11434/api
    e devolve SEMPRE o endpoint /api/chat.
    """
    host = (host or "").strip().rstrip("/")

    if not host:
        raise ValueError("OLLAMA_HOST está vazio. Configure no config.py ou via variável de ambiente.")

    # Já veio completo?
    if host.endswith("/api/chat"):
        return host

    # Veio como /api ?
    if host.endswith("/api"):
        return f"{host}/chat"

    # Caso padrão: acrescenta /api/chat
    return f"{host}/api/chat"


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: bool = False,
    timeout: int = 120,
    **kwargs,
) -> Dict[str, Any]:
    """
    Chama o endpoint de chat do Ollama.
    Retorna JSON (dict) conforme o servidor responder.
    """
    url = _normalize_chat_url(OLLAMA_HOST)
    body: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": bool(stream),
    }
    # permite overrides (ex.: options, format, keep_alive etc.)
    if kwargs:
        body.update(kwargs)

    r = requests.post(url, headers=HEADERS_JSON, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()
