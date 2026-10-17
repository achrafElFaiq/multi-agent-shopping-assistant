"""Tools for reading, asking about, and updating a single user's category preferences, and for reading past offers."""

import logging
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from backend.core.models.preferences import UserPreferences
from backend.core.models.recommendations import RecommendationRecord
from backend.core.ports.preferences import PreferenceRemoval, Preferences, PreferencesRepository
from backend.core.ports.recommendations import RecommendationRepository

logger = logging.getLogger(__name__)

State = Annotated[dict[str, Any], InjectedState]  # the agent's messages, given by the system, not the model

# Past offers are hints for one request: nothing read from them is saved, whatever the model does.
AFTER_HISTORY = "Preferences can't be changed after reading past offers: use them for this request only."
BEFORE_HISTORY = "Read past offers first: call search_history, then submit again with them as hints."


class Submission(UserPreferences):
    """What the model submits, plus the agent's messages, given by the system."""

    state: State


def make_preference_tools(
    repository: PreferencesRepository, recommendations: RecommendationRepository
) -> list[BaseTool]:
    @tool
    def get_preferences(category: str) -> dict[str, Any]:
        """Read stored preferences for a category, including general. Null preferences mean the category is unknown."""
        row = repository.get_preferences(category)
        return row.model_dump(mode="json") if row is not None else {"category": category, "preferences": None}

    @tool
    def search_history(query: str) -> list[dict[str, Any]]:
        """Find past offers for requests like this one, whether the shopper accepted them, and why not.

        query: what the shopper is looking for, in a few words, e.g. "trail running shoes".
        """
        return [past_answer(record) for record in recommendations.find_similar(query)]

    @tool
    def ask_user(question: str) -> Any:
        """Ask one concise question for essential missing preferences. Call this tool alone.

        The user's answer is returned to you; use save_preferences to save lasting facts.
        """
        return interrupt({"type": "preferences", "question": question})

    @tool
    def save_preferences(category: str, changes: Preferences, state: State) -> dict[str, Any] | str:
        """Save lasting preferences explicitly stated by the user. Send only what changes.

        List items are added to the stored ones; attributes and the budget are set. Nothing is removed.
        """
        if history_read(state):
            logger.info("refused a save after reading past offers: %s", changes.model_dump_json(exclude_defaults=True))
            return AFTER_HISTORY
        return repository.save_preferences(category, changes).model_dump(mode="json")

    @tool
    def remove_preferences(category: str, removal: PreferenceRemoval, state: State) -> dict[str, Any] | str:
        """Remove preferences the user says no longer apply: attribute keys, list items, or the budget.

        Only what is named is removed.
        """
        if history_read(state):
            logger.info(
                "refused a removal after reading past offers: %s", removal.model_dump_json(exclude_defaults=True)
            )
            return AFTER_HISTORY
        row = repository.remove_preferences(category, removal)
        return row.model_dump(mode="json") if row is not None else {"category": category, "preferences": None}

    # An accepted submission ends the agent. The model gets the text, the code gets the artifact.
    @tool(args_schema=Submission, return_direct=True, response_format="content_and_artifact")
    def submit_preferences(state: State, **preferences: Any) -> tuple[str, UserPreferences | None]:
        """Submit preferences for this shopping request. This ends your work without saving them to the database.

        The first call asks you to read past offers with search_history, then to submit again.
        """
        if not history_read(state):
            logger.info("asked to read past offers before submitting")
            return BEFORE_HISTORY, None
        return "Preferences received.", UserPreferences(**preferences)

    tools = [get_preferences, search_history, ask_user, save_preferences, remove_preferences, submit_preferences]
    # Invalid arguments become tool results so the model can correct them.
    for preference_tool in tools:
        preference_tool.handle_validation_error = lambda error: f"Invalid preferences: {error}"
    return tools


def history_read(state: dict[str, Any]) -> bool:
    return any(isinstance(message, ToolMessage) and message.name == "search_history" for message in state["messages"])


def past_answer(record: RecommendationRecord) -> dict[str, Any]:
    """What the agent needs from a past answer: what was offered, and whether and why the shopper said no."""
    return {
        "request": record.query,
        "product": record.offer.title,
        "store": record.offer.seller.name,
        "price": f"{record.offer.price} {record.offer.currency}",
        "options": record.offer.options,
        "accepted": record.accepted,
        "reason": record.reason,
        "date": record.decided_at.date().isoformat(),
    }
