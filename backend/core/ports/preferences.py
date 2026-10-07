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

    def update_preferences(self, category: str, changes: Preferences) -> CategoryPreferences:
        """Create or update a category and set updated_at to the current UTC time.

        Only fields in changes.model_dump(exclude_unset=True) are updated.
        Merge attributes by key; replace supplied lists and the supplied budget.
        Omitted fields keep their existing values.
        """
        ...
