"""In-memory PreferencesRepository with fixed data. Used until the Postgres adapter exists."""

from datetime import date
from decimal import Decimal

from backend.core.ports.preferences import PastRecommendation, Profile

MOCK_PROFILES = {
    "user-1": Profile(
        sizes={"shoes": "43", "t-shirt": "M"},
        budgets={"shoes": Decimal("130")},
        likes=["Asics", "black"],
        dislikes=["suede"],
    ),
}

MOCK_RECOMMENDATIONS = {
    "user-1": [
        PastRecommendation(
            category="shoes",
            product_name="Gel-Nimbus 25",
            brand="Asics",
            price=Decimal("119.90"),
            approved=True,
            reason="very comfortable",
            recommended_on=date(2026, 8, 3),
        ),
        PastRecommendation(
            category="shoes",
            product_name="Pegasus 41",
            brand="Nike",
            price=Decimal("129.99"),
            approved=False,
            reason="too narrow",
            recommended_on=date(2026, 6, 12),
        ),
        PastRecommendation(
            category="t-shirt",
            product_name="Dri-FIT Tee",
            brand="Nike",
            price=Decimal("29.99"),
            approved=True,
            recommended_on=date(2026, 5, 20),
        ),
    ],
}


class MockPreferencesRepository:
    def __init__(
        self,
        profiles: dict[str, Profile] | None = None,
        recommendations: dict[str, list[PastRecommendation]] | None = None,
    ) -> None:
        self._profiles = MOCK_PROFILES if profiles is None else profiles
        self._recommendations = MOCK_RECOMMENDATIONS if recommendations is None else recommendations

    def get_profile(self, user_id: str) -> Profile:
        return self._profiles.get(user_id, Profile())

    def get_recommendations(self, user_id: str) -> list[PastRecommendation]:
        return sorted(self._recommendations.get(user_id, []), key=lambda r: r.recommended_on, reverse=True)
