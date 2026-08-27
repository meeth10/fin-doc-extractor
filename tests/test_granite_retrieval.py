from __future__ import annotations

from typing import Any, Dict

import pytest

from agent.retrieval_agent import _candidate_index, retrieve


def _evidence() -> Dict[str, Any]:
    return {
        "metric": "total_debt",
        "intent": "value",
        "candidates": [
            {
                "metric": "total_debt", "page": 34, "statement": "balance_sheet",
                "table_title": "Consolidated Balance Sheets", "matched_alias": "total debt",
                "values": [1066.0, 958.0], "periods": ["2025", "2024"],
                "validated": True, "assignment": "balance_sheet", "score": 9.8,
            },
            {
                "metric": "total_debt", "page": 34, "statement": "balance_sheet",
                "table_title": "Consolidated Balance Sheets", "matched_alias": "commercial paper",
                "values": [0.0, 0.0], "periods": ["2025", "2024"],
                "validated": True, "assignment": "balance_sheet", "score": 9.7,
            },
            {
                "metric": "total_debt", "page": 34, "statement": "balance_sheet",
                "table_title": "Consolidated Balance Sheets", "matched_alias": "current term debt",
                "values": [12.0, 10.0], "periods": ["2025", "2024"],
                "validated": True, "assignment": "balance_sheet", "score": 9.6,
            },
            {
                "metric": "total_debt", "page": 34, "statement": "balance_sheet",
                "table_title": "Consolidated Balance Sheets", "matched_alias": "non-current term debt",
                "values": [1054.0, 948.0], "periods": ["2025", "2024"],
                "validated": True, "assignment": "balance_sheet", "score": 9.5,
            },
        ],
        "raw_evidence": [],
        "computed": None,
    }


def test_candidate_index_is_limited_and_preserves_identity():
    index = _candidate_index(_evidence())
    assert len(index) == 4
    assert {item["matched_alias"] for item in index} == {
        "total debt", "commercial paper", "current term debt", "non-current term debt"
    }
    assert all(item["page"] == 34 for item in index)


def test_retrieve_requires_valid_model_selection(monkeypatch):
    expected = _evidence()
    selected = _candidate_index(expected)

    def fake_chat(*args: Any, **kwargs: Any):
        return ({"selected_sources": selected[:4]}, '{"selected_sources": [ ... ]}')

    monkeypatch.setattr("agent.retrieval_agent.chat_json_with_trace", fake_chat)
    result = retrieve("What was total debt?", expected)

    assert result["fallback"] is False
    assert result["model"] == "ibm/granite4.2:3b"
    assert len(result["selected_sources"]) == 4
    assert result["raw_model_output"]


def test_retrieve_falls_back_when_granite_returns_unknown_source(monkeypatch):
    expected = _evidence()

    def fake_chat(*args: Any, **kwargs: Any):
        return ({"selected_sources": [{"metric": "total_debt", "page": 999}]}, '{"selected_sources":[{"metric":"total_debt","page":999}]}')

    monkeypatch.setattr("agent.retrieval_agent.chat_json_with_trace", fake_chat)
    result = retrieve("What was total debt?", expected)

    assert result["fallback"] is True
    assert result["warnings"]
    assert "valid candidate" in result["warnings"][0]
    assert result["raw_model_output"] is not None
    assert result["selected_sources"][0]["page"] == 34


def test_retrieve_falls_back_when_granite_json_is_invalid(monkeypatch):
    expected = _evidence()

    def fake_chat(*args: Any, **kwargs: Any):
        raise RuntimeError("Ollama returned invalid JSON.")

    monkeypatch.setattr("agent.retrieval_agent.chat_json_with_trace", fake_chat)
    result = retrieve("What was total debt?", expected)

    assert result["fallback"] is True
    assert result["raw_model_output"] is None
    assert "Granite retrieval fallback used" in result["warnings"][0]
    assert result["selected_sources"][0]["page"] == 34
