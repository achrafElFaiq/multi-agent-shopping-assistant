"""Mock RecommendationRepository: keeps the answers in memory and logs them. A PostgreSQL table replaces it later."""

import logging

from backend.core.models.recommendations import RecommendationRecord

logger = logging.getLogger(__name__)


class InMemoryRecommendations:
    def __init__(self) -> None:
        self.records: list[RecommendationRecord] = []

    def save(self, record: RecommendationRecord) -> None:
        self.records.append(record)
        if record.accepted:
            logger.info("[mock] saved: accepted %r from %s", record.offer.title, record.offer.seller.name)
        else:
            logger.info("[mock] saved: rejected %r, reason: %r", record.offer.title, record.reason or "none given")
