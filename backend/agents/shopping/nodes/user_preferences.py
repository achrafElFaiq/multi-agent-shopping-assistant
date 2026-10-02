"""Chapter 1: Personal Memory.

Stub: returns fixed preferences. Will be replaced by the real memory (PostgreSQL, Qdrant).
"""

from decimal import Decimal

from backend.agents.shopping.schemas import UserPreferences
from backend.agents.shopping.state import ShoppingState


def build_user_preferences(state: ShoppingState) -> dict[str, UserPreferences]:
    user_id = state["request"].user_id
    preferences = UserPreferences(
        user_id=user_id,
        category="shoes",
        summary="Prefers cushioned running shoes. Kept all Asics pairs; returned Nike as too narrow.",
        size="43",
        budget=Decimal("130"),
        likes=["Asics", "cushioned", "black"],
        avoid=["Nike (narrow fit)"],
    )
    return {"preferences": preferences}
