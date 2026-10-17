"""Single-user preferences stored in PostgreSQL through psycopg."""

from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from backend.core.ports.preferences import CategoryPreferences, PreferenceRemoval, Preferences
from backend.core.services.preferences import removed, saved


class PostgresPreferencesRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        with psycopg.connect(self._database_url) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    category TEXT PRIMARY KEY,
                    preferences JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
            """)

    def list_categories(self) -> list[str]:
        with psycopg.connect(self._database_url) as connection:
            rows = connection.execute("SELECT category FROM preferences ORDER BY category").fetchall()
            return [row[0] for row in rows]

    def get_preferences(self, category: str) -> CategoryPreferences | None:
        with psycopg.connect(self._database_url, row_factory=class_row(CategoryPreferences)) as connection:
            return connection.execute(
                "SELECT category, preferences, updated_at FROM preferences WHERE category = %s", (category,)
            ).fetchone()

    def save_preferences(self, category: str, changes: Preferences) -> CategoryPreferences:
        with psycopg.connect(self._database_url) as connection:
            stored = locked(connection, category)
            return write(connection, category, saved(stored or Preferences(), changes))

    def remove_preferences(self, category: str, removal: PreferenceRemoval) -> CategoryPreferences | None:
        with psycopg.connect(self._database_url) as connection:
            stored = locked(connection, category)
            return write(connection, category, removed(stored, removal)) if stored else None


def locked(connection: psycopg.Connection[tuple[Any, ...]], category: str) -> Preferences | None:
    """Read a category and lock its row until the end of the transaction, so changes apply to what is stored."""
    row = connection.execute(
        "SELECT preferences FROM preferences WHERE category = %s FOR UPDATE", (category,)
    ).fetchone()
    return Preferences.model_validate(row[0]) if row else None


def write(
    connection: psycopg.Connection[tuple[Any, ...]], category: str, preferences: Preferences
) -> CategoryPreferences:
    row = CategoryPreferences(category=category, preferences=preferences, updated_at=datetime.now(UTC))
    connection.execute(
        """
        INSERT INTO preferences (category, preferences, updated_at) VALUES (%s, %s, %s)
        ON CONFLICT (category) DO UPDATE SET preferences = EXCLUDED.preferences, updated_at = EXCLUDED.updated_at
        """,
        (row.category, Jsonb(row.preferences.model_dump(mode="json")), row.updated_at),
    )
    return row
