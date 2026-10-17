"""Single-user preferences: one row per category, with structured JSON and an update timestamp."""

from datetime import datetime
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

type AttributeValue = str | int | float | bool | list[str | int | float | bool]


class Budget(BaseModel):
    """Usual spending in one currency, separate from a purchase's spending limit."""

    model_config = ConfigDict(extra="forbid")

    amount: Decimal = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class Preferences(BaseModel):
    """The JSON stored for a category. Missing information stays empty or null."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    usual_budget: Budget | None = None


class PreferenceRemoval(BaseModel):
    """What to remove from a category: attribute keys, list items, and the budget if asked."""

    model_config = ConfigDict(extra="forbid")

    attributes: list[str] = Field(default_factory=list)  # keys, e.g. ["gender"]
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    usual_budget: bool = False


class CategoryPreferences(BaseModel):
    """One row in the preferences table, including the general category."""

    category: str = Field(min_length=1)
    preferences: Preferences
    updated_at: datetime


class PreferencesRepository(Protocol):
    """Category-based storage for one user, implemented by the PostgreSQL adapter."""

    def list_categories(self) -> list[str]:
        """Return existing category names in alphabetical order."""
        ...

    def get_preferences(self, category: str) -> CategoryPreferences | None:
        """Return the category's row, or None if it does not exist."""
        ...

    def save_preferences(self, category: str, changes: Preferences) -> CategoryPreferences:
        """Create or update a category and set updated_at to the current UTC time.

        List items are added to the stored ones, without duplicates (ignoring case).
        Attributes are set by key, and the budget is set if given. Nothing is removed.
        """
        ...

    def remove_preferences(self, category: str, removal: PreferenceRemoval) -> CategoryPreferences | None:
        """Remove the named attribute keys and list items (ignoring case), and the budget if asked.

        Return None if the category does not exist.
        """
        ...
