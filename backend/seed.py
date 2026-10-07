"""Add example preference categories: uv run python -m backend.seed.

Existing categories are preserved when the command is run again.
"""

from datetime import UTC, datetime
from decimal import Decimal

import psycopg
from psycopg.types.json import Jsonb

from backend.config.settings import load_settings
from backend.core.adapters.postgres_preferences import PostgresPreferencesRepository
from backend.core.ports.preferences import Budget, Preferences

SEED_PREFERENCES = {
    "general": Preferences(
        attributes={"country": "France", "preferred_currency": "EUR"},
        likes=["black", "simple design", "durable products"],
        dislikes=["suede", "large visible logos"],
        requirements=["clear return policy"],
    ),
    "shoes": Preferences(
        attributes={"size": "43 EU", "width": "wide", "activity": "running"},
        likes=["Asics", "comfort"],
        dislikes=["narrow fit"],
        requirements=["cushioning", "breathable upper"],
        usual_budget=Budget(amount=Decimal("130"), currency="EUR"),
    ),
    "t-shirt": Preferences(
        attributes={"size": "M", "fit": "regular", "colors": ["black", "white", "navy"]},
        likes=["cotton", "plain designs"],
        dislikes=["large prints", "tight fit"],
        requirements=["machine washable"],
        usual_budget=Budget(amount=Decimal("35"), currency="EUR"),
    ),
    "jeans": Preferences(
        attributes={"size": "W32 L32", "fit": "straight"},
        likes=["dark blue", "stretch denim"],
        dislikes=["skinny fit", "distressed fabric"],
        requirements=["comfortable when sitting"],
        usual_budget=Budget(amount=Decimal("80"), currency="EUR"),
    ),
    "jacket": Preferences(
        attributes={"size": "M", "season": ["autumn", "winter"]},
        likes=["lightweight materials", "zippered pockets"],
        dislikes=["bulky designs", "fur"],
        requirements=["water resistant", "hood"],
        usual_budget=Budget(amount=Decimal("160"), currency="EUR"),
    ),
    "laptop": Preferences(
        attributes={"os": "Linux", "screen_inches": 14, "min_ram_gb": 16, "min_storage_gb": 512},
        likes=["long battery life", "lightweight design"],
        dislikes=["loud fans", "glossy screens"],
        requirements=["Linux compatibility", "USB-C charging", "comfortable keyboard"],
        usual_budget=Budget(amount=Decimal("1000"), currency="EUR"),
    ),
    "headphones": Preferences(
        attributes={"style": "over-ear", "wireless": True},
        likes=["noise cancellation", "soft ear cushions"],
        dislikes=["heavy bass", "tight clamping force"],
        requirements=["multipoint Bluetooth", "replaceable ear pads"],
        usual_budget=Budget(amount=Decimal("180"), currency="EUR"),
    ),
    "smartphone": Preferences(
        attributes={"os": "Android", "min_storage_gb": 128, "max_screen_inches": 6.5},
        likes=["long battery life", "long software support"],
        dislikes=["curved screens", "preinstalled advertising apps"],
        requirements=["NFC", "USB-C charging"],
        usual_budget=Budget(amount=Decimal("450"), currency="EUR"),
    ),
    "backpack": Preferences(
        attributes={"capacity_liters": 24, "laptop_inches": 14, "use": ["work", "travel"]},
        likes=["simple design", "water resistant fabric"],
        dislikes=["leather", "too many external straps"],
        requirements=["padded laptop compartment", "comfortable shoulder straps"],
        usual_budget=Budget(amount=Decimal("90"), currency="EUR"),
    ),
    "office-chair": Preferences(
        attributes={"room": "home office", "daily_use_hours": 8},
        likes=["breathable mesh", "adjustable armrests"],
        dislikes=["faux leather", "gaming chair designs"],
        requirements=["adjustable lumbar support", "adjustable seat height"],
        usual_budget=Budget(amount=Decimal("250"), currency="EUR"),
    ),
}


def seed(database_url: str) -> int:
    """Create the table if needed and return the number of inserted categories."""
    PostgresPreferencesRepository(database_url)
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO preferences (category, preferences, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (category) DO NOTHING
            """,
            [
                (category, Jsonb(preferences.model_dump(mode="json")), datetime.now(UTC))
                for category, preferences in SEED_PREFERENCES.items()
            ],
        )
        return cursor.rowcount


def main() -> None:
    settings = load_settings()
    inserted = seed(settings.database_url.get_secret_value())
    print(f"Added {inserted} preference categories.")


if __name__ == "__main__":
    main()
