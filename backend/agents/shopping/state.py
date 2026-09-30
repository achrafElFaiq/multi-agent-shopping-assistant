"""State shared by all nodes of the shopping graph.

The graph starts with only `request`. Each node adds its own field.
"""

from typing import NotRequired, TypedDict

from backend.agents.shopping.schemas import (
    ApprovalDecision,
    PaymentResult,
    Recommendation,
    ShoppingRequest,
    UserPreferences,
)


class ShoppingState(TypedDict):
    request: ShoppingRequest
    preferences: NotRequired[UserPreferences]  # set by user_preferences
    recommendation: NotRequired[Recommendation]  # set by product_search
    decision: NotRequired[ApprovalDecision]  # set by approval
    payment: NotRequired[PaymentResult]  # set by payment
