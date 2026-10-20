"""Tools of the preferences agent, in two sets that never mix.

Saving: read, ask and save what the shopper states, without past offers.
Preparing: read past offers and submit the preferences for the request, with no way to save anything.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from backend.core.models.preferences import UserPreferences
from backend.core.models.recommendations import RecommendationRecord
from backend.core.ports.preferences import PreferenceRemoval, Preferences, PreferencesRepository
from backend.core.ports.recommendations import RecommendationRepository


def make_saving_tools(repository: PreferencesRepository) -> list[BaseTool]:
    @tool
    def ask_user(question: str) -> Any:
        """Ask one concise question for essential missing preferences. Call this tool alone.

        The user's answer is returned to you; use save_preferences to save lasting facts.
        """
        return interrupt({"type": "preferences", "question": question})

    @tool
    def save_preferences(category: str, changes: Preferences) -> dict[str, Any]:
        """Save lasting preferences explicitly stated by the user. Send only what changes.

        List items are added to the stored ones; attributes and the budget are set. Nothing is removed.
        """
        return repository.save_preferences(category, changes).model_dump(mode="json")

    @tool
    def remove_preferences(category: str, removal: PreferenceRemoval) -> dict[str, Any]:
        """Remove preferences the user says no longer apply: attribute keys, list items, or the budget.

        Only what is named is removed.
        """
        row = repository.remove_preferences(category, removal)
        return row.model_dump(mode="json") if row is not None else {"category": category, "preferences": None}

    @tool(response_format="content_and_artifact")
    def done(category: str) -> tuple[str, str]:
        """Call this alone when what the shopper stated is saved, or when nothing needed saving.

        category: the category chosen for this request.
        """
        return "Done.", category  # the code gets the category, for the preparing step

    return checked([get_preferences_tool(repository), ask_user, save_preferences, remove_preferences, done])


def make_preparing_tools(
    repository: PreferencesRepository, recommendations: RecommendationRepository
) -> list[BaseTool]:
    @tool
    def search_history(query: str) -> list[dict[str, Any]]:
        """Find past offers for requests like this one, whether the shopper accepted them, and why not.

        query: what the shopper is looking for, in a few words, e.g. "trail running shoes".
        """
        return [past_answer(record) for record in recommendations.find_similar(query)]

    # Submitting ends the agent. The model gets the text, the code gets the artifact.
    @tool(args_schema=UserPreferences, return_direct=True, response_format="content_and_artifact")
    def submit_preferences(**preferences: Any) -> tuple[str, UserPreferences]:
        """Submit preferences for this shopping request. This ends your work without saving them to the database."""
        return "Preferences received.", UserPreferences(**preferences)

    return checked([get_preferences_tool(repository), search_history, submit_preferences])


def get_preferences_tool(repository: PreferencesRepository) -> BaseTool:
    @tool
    def get_preferences(category: str) -> dict[str, Any]:
        """Read stored preferences for a category, including general. Null preferences mean the category is unknown."""
        row = repository.get_preferences(category)
        return row.model_dump(mode="json") if row is not None else {"category": category, "preferences": None}

    return get_preferences


def checked(tools: list[BaseTool]) -> list[BaseTool]:
    """Invalid arguments become tool results so the model can correct them."""
    for preference_tool in tools:
        preference_tool.handle_validation_error = lambda error: f"Invalid preferences: {error}"
    return tools


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
