"""The mock preferences repository follows the same rules as the PostgreSQL one."""

from backend.core.adapters.memory_preferences import InMemoryPreferencesRepository
from backend.core.ports.preferences import PreferenceRemoval, Preferences


def test_reads_the_preferences_it_starts_with() -> None:
    repository = InMemoryPreferencesRepository({"shoes": Preferences(likes=["Asics"]), "general": Preferences()})

    assert repository.list_categories() == ["general", "shoes"]
    shoes = repository.get_preferences("shoes")
    assert shoes is not None and shoes.preferences.likes == ["Asics"]
    assert repository.get_preferences("coffee") is None


def test_saves_and_removes_without_touching_the_rest() -> None:
    repository = InMemoryPreferencesRepository(
        {"shoes": Preferences(attributes={"size": "43 EU", "width": "wide"}, likes=["Asics"], dislikes=["narrow fit"])}
    )

    saved = repository.save_preferences("shoes", Preferences(attributes={"size": "44 EU"}, dislikes=["suede"]))

    assert saved.preferences.attributes == {"size": "44 EU", "width": "wide"}
    assert saved.preferences.likes == ["Asics"]
    assert saved.preferences.dislikes == ["narrow fit", "suede"]
    assert repository.get_preferences("shoes") == saved

    after = repository.remove_preferences("shoes", PreferenceRemoval(attributes=["width"], dislikes=["Narrow Fit"]))

    assert after is not None and after.preferences.attributes == {"size": "44 EU"}
    assert after.preferences.dislikes == ["suede"]
    assert repository.remove_preferences("coffee", PreferenceRemoval(likes=["light roast"])) is None


def test_creates_a_new_category() -> None:
    repository = InMemoryPreferencesRepository()

    repository.save_preferences("coffee", Preferences(likes=["light roast"]))

    coffee = repository.get_preferences("coffee")
    assert coffee is not None and coffee.preferences.likes == ["light roast"]
