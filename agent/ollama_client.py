from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_CONTEXT = int(os.getenv("OLLAMA_CONTEXT", "32768"))
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "768"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))


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
        try:
            value = json.loads(fenced.group(1))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Ollama JSON response must be an object.")
    return value


def _request_json(system: str, user: str, model: str, think: bool | str, num_ctx: int, num_predict: int) -> Dict[str, Any]:
    parsed, _ = _request_json_with_raw(system, user, model, think, num_ctx, num_predict)
    return parsed


def _request_json_with_raw(system: str, user: str, model: str, think: bool | str, num_ctx: int, num_predict: int) -> tuple[Dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "think": think,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"Ollama returned HTTP {exc.code} at {OLLAMA_BASE_URL}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {OLLAMA_BASE_URL}. Start it with `ollama serve`.") from exc
    content = body.get("message", {}).get("content", "")
    parsed = _decode_json_content(content)
    return parsed, str(content)


def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    *,
    think: bool | str = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
) -> Dict[str, Any]:
    selected_model = model or OLLAMA_MODEL
    context = num_ctx or OLLAMA_CONTEXT
    output = num_predict or OLLAMA_MAX_OUTPUT
    try:
        return _request_json(system, user, selected_model, think, context, output)
    except RuntimeError as exc:
        if "invalid JSON" not in str(exc):
            raise
        retry_system = f"{system}\n\nCRITICAL OUTPUT CONTRACT: return exactly one valid JSON object. No markdown fences, no commentary, no leading or trailing text."
        return _request_json(retry_system, user, selected_model, think, context, output)


def chat_json_with_trace(
    system: str,
    user: str,
    model: str | None = None,
    *,
    think: bool | str = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
) -> tuple[Dict[str, Any], str]:
    """Return parsed JSON plus the exact model content for debugging/auditing."""
    selected_model = model or OLLAMA_MODEL
    context = num_ctx or OLLAMA_CONTEXT
    output = num_predict or OLLAMA_MAX_OUTPUT
    try:
        return _request_json_with_raw(system, user, selected_model, think, context, output)
    except RuntimeError as exc:
        if "invalid JSON" not in str(exc):
            raise
        retry_system = f"{system}\n\nCRITICAL OUTPUT CONTRACT: return exactly one valid JSON object. No markdown fences, no commentary, no leading or trailing text."
        return _request_json_with_raw(retry_system, user, selected_model, think, context, output)
