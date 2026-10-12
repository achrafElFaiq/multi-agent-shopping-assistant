"""Asking the preferences agent for the user's preferences, implemented over A2A."""

from collections.abc import Callable
from typing import Any, Protocol

from backend.core.models.preferences import UserPreferences
from backend.core.models.shopping import ShoppingRequest


class PreferencesAgent(Protocol):
    def get_preferences(
        self, request: ShoppingRequest, conversation_id: str, ask_user: Callable[[dict[str, Any]], Any]
    ) -> UserPreferences:
        """The user's preferences for this request.

        The agent may ask the user questions: each one goes to `ask_user`, which returns the answer.
        `conversation_id` is the same each time the same request is asked again, e.g. after a pause.
        """
        ...
