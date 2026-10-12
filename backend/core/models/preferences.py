"""What the preferences agent answers: the user's preferences for one request."""

from pydantic import Field

from backend.core.ports.preferences import Preferences


class UserPreferences(Preferences):
    """Preferences for the shopping request, with its category and a summary."""

    category: str = Field(min_length=1)
    summary: str
