from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    number: int
    text: str
    extraction_method: str
    text_quality: float
    statement_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StatementRegion:
    statement_type: str
    start_page: int
    end_page: int
    confidence: float
    evidence_pages: list[int] = field(default_factory=list)


@dataclass
class TableEvidence:
    page: int
    rows: list[list[str]]
    statement_type: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    extraction_method: str = "unknown"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceDocument:
    source_file: str
    page_count: int
    metadata: dict[str, Any]
    pages: list[Page]
    statement_regions: list[StatementRegion]
    tables: list[TableEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "pages": [vars(page) for page in self.pages],
            "statement_regions": [vars(region) for region in self.statement_regions],
            "tables": [vars(table) for table in self.tables],
        }
