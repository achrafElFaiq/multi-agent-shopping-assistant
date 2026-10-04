"""Tools the user preferences agent can call.

They are built for one user: the code fixes `user_id`, so the model can never read another user's data.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool

from backend.agents.shopping.schemas import UserPreferences
from backend.core.ports.preferences import PreferencesRepository


def make_preference_tools(repository: PreferencesRepository, user_id: str) -> list[BaseTool]:
    @tool
    def get_profile() -> dict[str, Any]:
        """Get what the user told us about themselves: sizes and usual budget per category, likes and dislikes."""
        return repository.get_profile(user_id).model_dump(mode="json")

    @tool
    def get_recommendations() -> list[dict[str, Any]]:
        """Get every product we recommended to the user before, in all categories, most recent first. Each one says whether the user approved it and why."""  # noqa: E501
        return [r.model_dump(mode="json") for r in repository.get_recommendations(user_id)]

    # return_direct: calling this tool ends the agent. The model gets the text, the code gets the artifact.
    @tool(args_schema=UserPreferences, return_direct=True, response_format="content_and_artifact")
    def submit_preferences(**preferences: Any) -> tuple[str, UserPreferences]:
        """Give the user's preferences for this request, once you know enough. This ends your work."""
        return "Preferences received.", UserPreferences(**preferences)

    # Invalid preferences are sent back to the model as the tool result, so it can fix them.
    submit_preferences.handle_validation_error = lambda error: f"Invalid preferences: {error}"

    return [get_profile, get_recommendations, submit_preferences]
