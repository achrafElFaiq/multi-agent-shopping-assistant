"""Keeping the shopper's answers to recommendations, implemented in memory for now (PostgreSQL later)."""

from typing import Protocol

from backend.core.models.recommendations import RecommendationRecord


class RecommendationRepository(Protocol):
    def save(self, record: RecommendationRecord) -> None:
        """Keep one answer: an accepted offer, or a rejected one with the reason."""
        ...
