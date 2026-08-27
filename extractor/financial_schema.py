from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceRef:
    page: Optional[int] = None
    statement: Optional[str] = None
    table_title: Optional[str] = None
    source: Optional[str] = None
    fact_id: Optional[str] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "statement": self.statement,
            "table_title": self.table_title,
            "source": self.source,
            "fact_id": self.fact_id,
            "row_index": self.row_index,
            "column_index": self.column_index,
        }


@dataclass
class FinancialAnswer:
    metric: str
    answer: Optional[float | str]
    period: Optional[str]
    currency: Optional[str]
    unit: Optional[str]
    status: str  # reported | derived | reconstructed | inferred | ambiguous | not_available
    confidence: str  # high | medium | low
    formula: Optional[str] = None
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[EvidenceRef] = field(default_factory=list)
    explanation: Optional[str] = None
    scope: Optional[str] = None
    definition: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    verification: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "answer": self.answer,
            "period": self.period,
            "currency": self.currency,
            "unit": self.unit,
            "status": self.status,
            "confidence": self.confidence,
            "formula": self.formula,
            "inputs": self.inputs,
            "sources": [s.as_dict() for s in self.sources],
            "explanation": self.explanation,
            "scope": self.scope,
            "definition": self.definition,
            "warnings": self.warnings,
            "verification": self.verification,
        }
