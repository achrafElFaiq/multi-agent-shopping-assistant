"""The mock preferences repository follows the same rules as the PostgreSQL one."""

from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.ports.preferences import Preferences


def test_reads_the_preferences_it_starts_with() -> None:
    repository = InMemoryPreferencesRepository({"shoes": Preferences(likes=["Asics"]), "general": Preferences()})

    assert repository.list_categories() == ["general", "shoes"]
    shoes = repository.get_preferences("shoes")
    assert shoes is not None and shoes.preferences.likes == ["Asics"]
    assert repository.get_preferences("coffee") is None


def test_partial_update_merges_attributes_and_replaces_supplied_lists() -> None:
    repository = InMemoryPreferencesRepository(
        {"shoes": Preferences(attributes={"size": "43 EU", "width": "wide"}, likes=["Asics"], dislikes=["suede"])}
    )

    saved = repository.update_preferences("shoes", Preferences(attributes={"size": "44 EU"}, likes=[]))

    assert saved.preferences.attributes == {"size": "44 EU", "width": "wide"}
    assert saved.preferences.likes == []
    assert saved.preferences.dislikes == ["suede"]
    assert repository.get_preferences("shoes") == saved


def test_creates_a_new_category() -> None:
    repository = InMemoryPreferencesRepository()

    repository.update_preferences("coffee", Preferences(likes=["light roast"]))

    coffee = repository.get_preferences("coffee")
    assert coffee is not None and coffee.preferences.likes == ["light roast"]
