"""The mock recommendation history: no meaning, just the latest answers."""

from datetime import UTC, datetime

from backend.core.adapters.memory_recommendations import InMemoryRecommendations
from backend.core.models.recommendations import RecommendationRecord
from tests.fakes import make_offer


def test_finds_the_latest_answers_newest_first() -> None:
    history = InMemoryRecommendations()
    for title in ["Nike Pegasus 42", "Asics Gel-Nimbus 26", "Hoka Clifton 10"]:
        history.save(
            RecommendationRecord(
                query="running shoes",
                category="shoes",
                offer=make_offer(title=title),
                pitch="",
                accepted=False,
                decided_at=datetime.now(UTC),
            )
        )

    found = history.find_similar("anything", limit=2)

    assert [record.offer.title for record in found] == ["Hoka Clifton 10", "Asics Gel-Nimbus 26"]
