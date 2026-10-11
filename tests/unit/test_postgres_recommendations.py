"""Answers to recommendations kept in PostgreSQL and found again by meaning, with vectors chosen by the test."""

from datetime import UTC, datetime

from backend.core.adapters.postgres_recommendations import PostgresRecommendations
from backend.core.models.recommendations import RecommendationRecord
from tests.fakes import FakeEmbedder, make_offer

VECTORS = {
    "running shoes": [1.0, 0.0, 0.0],
    "running shoes. Nike Pegasus 42": [1.0, 0.0, 0.0],
    "trail shoes. Salomon Speedcross 6": [0.8, 0.6, 0.0],  # cosine distance 0.2 from running shoes
    "coffee beans. Ethiopia light roast": [0.0, 0.0, 1.0],  # distance 1: unrelated
}


def answer(query: str, title: str, accepted: bool = False, reason: str = "") -> RecommendationRecord:
    return RecommendationRecord(
        query=query,
        category="shoes",
        offer=make_offer(title=title),
        pitch="Light and cushioned, as you like.",
        accepted=accepted,
        reason=reason,
        decided_at=datetime.now(UTC),
    )


def test_saved_answer_comes_back_whole(database_url: str) -> None:
    history = PostgresRecommendations(database_url, FakeEmbedder(VECTORS))
    refused = answer("running shoes", "Nike Pegasus 42", reason="too flashy")

    history.save(refused)

    assert history.find_similar("running shoes") == [refused]


def test_finds_close_answers_closest_first_and_leaves_out_unrelated_ones(database_url: str) -> None:
    history = PostgresRecommendations(database_url, FakeEmbedder(VECTORS))
    history.save(answer("coffee beans", "Ethiopia light roast", accepted=True))
    history.save(answer("trail shoes", "Salomon Speedcross 6", accepted=True))
    history.save(answer("running shoes", "Nike Pegasus 42", reason="too flashy"))

    found = history.find_similar("running shoes")

    assert [record.offer.title for record in found] == ["Nike Pegasus 42", "Salomon Speedcross 6"]
    assert [record.offer.title for record in history.find_similar("running shoes", limit=1)] == ["Nike Pegasus 42"]
