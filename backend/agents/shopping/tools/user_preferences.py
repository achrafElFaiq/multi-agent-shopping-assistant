"""Tools for reading, asking about, and updating a single user's category preferences."""

from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from backend.agents.shopping.schemas import UserPreferences
from backend.core.ports.preferences import Preferences, PreferencesRepository


def make_preference_tools(repository: PreferencesRepository) -> list[BaseTool]:
    @tool
    def get_preferences(category: str) -> dict[str, Any]:
        """Read stored preferences for a category, including general. Null preferences mean the category is unknown."""
        row = repository.get_preferences(category)
        return row.model_dump(mode="json") if row is not None else {"category": category, "preferences": None}

    @tool
    def ask_user(question: str) -> Any:
        """Ask one concise question for essential missing preferences. Call this tool alone.

        The user's answer is returned to you; use update_preferences to save lasting facts.
        """
        return interrupt({"type": "preferences", "question": question})

    @tool
    def update_preferences(category: str, changes: Preferences) -> dict[str, Any]:
        """Save lasting preferences explicitly stated by the user. Send only changed fields.

        Attribute keys merge; supplied lists and the budget replace their previous values.
        Read the category first and preserve list entries that still apply.
        """
        return repository.update_preferences(category, changes).model_dump(mode="json")

    # return_direct: calling this tool ends the agent. The model gets the text, the code gets the artifact.
    @tool(args_schema=UserPreferences, return_direct=True, response_format="content_and_artifact")
    def submit_preferences(**preferences: Any) -> tuple[str, UserPreferences]:
        """Submit preferences for this shopping request. This ends your work without saving them to the database."""
        return "Preferences received.", UserPreferences(**preferences)

    tools = [get_preferences, ask_user, update_preferences, submit_preferences]
    # Invalid arguments become tool results so the model can correct them.
    for preference_tool in tools:
        preference_tool.handle_validation_error = lambda error: f"Invalid preferences: {error}"
    return tools
