"""Compatibility facade for the Qwen retrieval stack."""

from .qwen_retrieval import EMBEDDING_MODEL, DOCUMENT_MODEL, retrieve

__all__ = ["EMBEDDING_MODEL", "DOCUMENT_MODEL", "retrieve"]
