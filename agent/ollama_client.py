from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_CONTEXT = int(os.getenv("OLLAMA_CONTEXT", "12288"))
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "768"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")


def _decode_json_content(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S | re.I)
        if not fenced:
            raise RuntimeError("Ollama returned invalid JSON.")
        value = json.loads(fenced.group(1))
    if not isinstance(value, dict):
        raise RuntimeError("Ollama JSON response must be an object.")
    return value


def _post(path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code} at {OLLAMA_BASE_URL}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama is unavailable at {OLLAMA_BASE_URL}. Start it with `ollama serve`."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned a malformed HTTP JSON response.") from exc


def _request_json_with_raw(
    system: str,
    user: str,
    model: str,
    think: bool | str,
    num_ctx: int,
    num_predict: int,
    format_schema: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": format_schema or "json",
        "think": think,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    body = _post("/api/chat", payload, OLLAMA_TIMEOUT)
    content = body.get("message", {}).get("content", "")
    return _decode_json_content(content), str(content)


def chat_json_with_trace(
    system: str,
    user: str,
    model: str | None = None,
    *,
    think: bool | str = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    format_schema: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], str]:
    selected_model = model or OLLAMA_MODEL
    context = num_ctx or OLLAMA_CONTEXT
    output = num_predict or OLLAMA_MAX_OUTPUT
    try:
        return _request_json_with_raw(
            system, user, selected_model, think, context, output, format_schema
        )
    except RuntimeError as exc:
        message = str(exc)
        if "invalid JSON" not in message:
            raise
        retry_system = (
            f"{system}\n\nCRITICAL OUTPUT CONTRACT: return exactly one valid JSON object "
            "and no markdown fences."
        )
        return _request_json_with_raw(
            retry_system, user, selected_model, think, context, output, format_schema
        )


def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    *,
    think: bool | str = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    format_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return chat_json_with_trace(
        system,
        user,
        model=model,
        think=think,
        num_ctx=num_ctx,
        num_predict=num_predict,
        format_schema=format_schema,
    )[0]


def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    if not texts:
        return []
    selected_model = model or OLLAMA_EMBED_MODEL
    body = _post(
        "/api/embed",
        {
            "model": selected_model,
            "input": texts,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
        OLLAMA_TIMEOUT,
    )
    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 0} "
            f"embeddings for {len(texts)} inputs."
        )
    return [[float(v) for v in row] for row in embeddings]
