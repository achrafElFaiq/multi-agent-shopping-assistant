"""Chapter 1: Personal Memory.

Stub: returns fixed preferences. Will be replaced by the real memory (PostgreSQL, Qdrant).
"""

from decimal import Decimal

from backend.agents.shopping.schemas import UserPreferences
from backend.agents.shopping.state import ShoppingState


def load_preferences(state: ShoppingState) -> dict[str, UserPreferences]:
    user_id = state["request"].user_id
    preferences = UserPreferences(
        user_id=user_id,
        sizes={"shoes": "43"},
        colours=["black", "blue"],
        brands=["Asics", "Nike"],
        max_price=Decimal("150"),
    )
    return {"preferences": preferences}
