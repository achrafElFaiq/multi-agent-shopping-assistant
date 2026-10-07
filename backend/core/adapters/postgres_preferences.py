"""Single-user preferences stored in PostgreSQL through psycopg."""

from datetime import UTC, datetime

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from backend.core.ports.preferences import CategoryPreferences, Preferences


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

    def update_preferences(self, category: str, changes: Preferences) -> CategoryPreferences:
        candidate = CategoryPreferences(category=category, preferences=changes, updated_at=datetime.now(UTC))
        patch = changes.model_dump(mode="json", exclude_unset=True)
        with psycopg.connect(self._database_url, row_factory=class_row(CategoryPreferences)) as connection:
            updated = connection.execute(
                """
                INSERT INTO preferences AS stored (category, preferences, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (category) DO UPDATE SET
                    preferences = (stored.preferences || %s) || jsonb_build_object(
                        'attributes', (stored.preferences -> 'attributes') || %s
                    ),
                    updated_at = EXCLUDED.updated_at
                RETURNING category, preferences, updated_at
                """,
                (
                    candidate.category,
                    Jsonb(candidate.preferences.model_dump(mode="json")),
                    candidate.updated_at,
                    Jsonb(patch),
                    Jsonb(patch.get("attributes", {})),
                ),
            ).fetchone()
            if updated is None:
                raise RuntimeError("PostgreSQL did not return the updated preferences.")
            return updated
