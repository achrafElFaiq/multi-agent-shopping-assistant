"""Mock PreferencesRepository: preferences kept in memory, forgotten when the program stops.

Starts with the example preferences of backend/seed.py. The PostgreSQL adapter replaces it for real use.
"""

from datetime import UTC, datetime

from backend.core.ports.preferences import CategoryPreferences, Preferences


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

    def update_preferences(self, category: str, changes: Preferences) -> CategoryPreferences:
        """Same rules as the PostgreSQL adapter: attributes merge by key, supplied lists and budget replace."""
        current = self._rows.get(category)
        stored = current.preferences.model_dump() if current else {}
        patch = changes.model_dump(exclude_unset=True)
        merged = {**stored, **patch}
        if "attributes" in patch:
            merged["attributes"] = {**stored.get("attributes", {}), **patch["attributes"]}
        row = CategoryPreferences(
            category=category, preferences=Preferences.model_validate(merged), updated_at=datetime.now(UTC)
        )
        self._rows[category] = row
        return row
