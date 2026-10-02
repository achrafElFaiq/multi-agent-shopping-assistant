"""Tools the user preferences agent can call.

They are built for one user: the code fixes `user_id`, so the model can never read another user's data.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool

from backend.core.ports.preferences import PreferencesRepository


def make_preference_tools(repository: PreferencesRepository, user_id: str) -> list[BaseTool]:
    @tool
    def get_profile() -> dict[str, Any]:
        """Get what the user told us about themselves: sizes and usual budget per category, likes and dislikes."""
        return repository.get_profile(user_id).model_dump(mode="json")

    @tool
    def get_recommendations(category: str) -> list[dict[str, Any]]:
        """Get products we recommended to the user before in a category such as "shoes", most recent first.

        Each one says whether the user approved it and why.
        """
        return [r.model_dump(mode="json") for r in repository.get_recommendations(user_id, category)]

    return [get_profile, get_recommendations]
