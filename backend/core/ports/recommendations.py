"""Keeping the shopper's answers to recommendations, and finding the past ones that matter for a new request."""

from typing import Protocol

from backend.core.models.recommendations import RecommendationRecord


class RecommendationRepository(Protocol):
    def save(self, record: RecommendationRecord) -> None:
        """Keep one answer: an accepted offer, or a rejected one with the reason."""
        ...

    def find_similar(self, query: str, limit: int = 5) -> list[RecommendationRecord]:
        """Return the past answers closest to a shopping request, closest first. Unrelated ones are left out."""
        ...
