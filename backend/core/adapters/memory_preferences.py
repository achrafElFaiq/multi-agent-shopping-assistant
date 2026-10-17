"""Mock PreferencesRepository: preferences kept in memory, forgotten when the program stops.

Starts with the example preferences of backend/seed.py. The PostgreSQL adapter replaces it for real use.
"""

from datetime import UTC, datetime

from backend.core.ports.preferences import CategoryPreferences, PreferenceRemoval, Preferences
from backend.core.services.preferences import removed, saved


class InMemoryPreferencesRepository:
    def __init__(self, preferences: dict[str, Preferences] | None = None) -> None:
        now = datetime.now(UTC)
        self._rows = {
            category: CategoryPreferences(category=category, preferences=stored, updated_at=now)
            for category, stored in (preferences or {}).items()
        }

    def list_categories(self) -> list[str]:
        return sorted(self._rows)

    def get_preferences(self, category: str) -> CategoryPreferences | None:
        return self._rows.get(category)

    def save_preferences(self, category: str, changes: Preferences) -> CategoryPreferences:
        current = self._rows.get(category)
        return self._store(category, saved(current.preferences if current else Preferences(), changes))

    def remove_preferences(self, category: str, removal: PreferenceRemoval) -> CategoryPreferences | None:
        current = self._rows.get(category)
        return self._store(category, removed(current.preferences, removal)) if current else None

    def _store(self, category: str, preferences: Preferences) -> CategoryPreferences:
        row = CategoryPreferences(category=category, preferences=preferences, updated_at=datetime.now(UTC))
        self._rows[category] = row
        return row
