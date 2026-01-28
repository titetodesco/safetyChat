
from __future__ import annotations
import requests, json
from ..config import OLLAMA_HOST, OLLAMA_MODEL, HEADERS_JSON

def chat(messages, model: str|None=None, stream: bool=False, timeout: int=120):
    body = {"model": model or OLLAMA_MODEL, "messages": messages, "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/chat", headers=HEADERS_JSON, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()
