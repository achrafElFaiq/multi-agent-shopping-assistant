"""Answers to recommendations stored in PostgreSQL, each with a vector (pgvector) to find those close to a request."""

import logging

import psycopg
from psycopg import sql
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from backend.core.models.recommendations import RecommendationRecord
from backend.core.ports.embeddings import Embedder

MAX_DISTANCE = 0.6  # cosine distance: 0 means the same thing, 1 unrelated. To be tuned on real requests.

logger = logging.getLogger(__name__)


class PostgresRecommendations:
    def __init__(self, database_url: str, embedder: Embedder) -> None:
        self._database_url = database_url
        self._embedder = embedder
        with psycopg.connect(self._database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            connection.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        query TEXT NOT NULL,
                        category TEXT NOT NULL,
                        offer JSONB NOT NULL,
                        pitch TEXT NOT NULL,
                        accepted BOOLEAN NOT NULL,
                        reason TEXT NOT NULL,
                        decided_at TIMESTAMPTZ NOT NULL,
                        embedding vector({dimensions}) NOT NULL
                    )
                """).format(dimensions=embedder.dimensions)
            )

    def save(self, record: RecommendationRecord) -> None:
        # A new request is compared to what was asked and what was offered.
        vector = self._embedder.embed(f"{record.query}. {record.offer.title}")
        with psycopg.connect(self._database_url) as connection:
            connection.execute(
                """
                INSERT INTO recommendations (query, category, offer, pitch, accepted, reason, decided_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    record.query,
                    record.category,
                    Jsonb(record.offer.model_dump(mode="json")),
                    record.pitch,
                    record.accepted,
                    record.reason,
                    record.decided_at,
                    str(vector),
                ),
            )
        if record.accepted:
            logger.info("saved: accepted %r from %s", record.offer.title, record.offer.seller.name)
        else:
            logger.info("saved: rejected %r, reason: %r", record.offer.title, record.reason)

    def find_similar(self, query: str, limit: int = 5) -> list[RecommendationRecord]:
        vector = str(self._embedder.embed(query))
        with psycopg.connect(self._database_url, row_factory=class_row(RecommendationRecord)) as connection:
            found = connection.execute(
                """
                SELECT query, category, offer, pitch, accepted, reason, decided_at
                FROM recommendations
                WHERE embedding <=> %(vector)s::vector <= %(max_distance)s
                ORDER BY embedding <=> %(vector)s::vector
                LIMIT %(limit)s
                """,
                {"vector": vector, "max_distance": MAX_DISTANCE, "limit": limit},
            ).fetchall()
        answers = [f"{r.offer.title} ({'yes' if r.accepted else 'no: ' + r.reason})" for r in found]
        logger.info("history for %r: %s", query, "; ".join(answers) or "nothing close")
        return found
