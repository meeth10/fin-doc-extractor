from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_CONTEXT = int(os.getenv("OLLAMA_CONTEXT", "16384"))
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "512"))


def chat_json(system: str, user: str, model: str | None = None) -> Dict[str, Any]:
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_ctx": OLLAMA_CONTEXT,
            "num_predict": OLLAMA_MAX_OUTPUT,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code} at {OLLAMA_BASE_URL}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama is unavailable at {OLLAMA_BASE_URL}. Start it with `ollama serve`."
        ) from exc
    content = body.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc
